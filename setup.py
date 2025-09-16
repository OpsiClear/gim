from setuptools import setup, find_packages

setup(
    packages=find_packages(include=['hloc', 'hloc.*', 'trainer', 'trainer.*', 'datasets', 'datasets.*', 'networks', 'networks.*']),
    py_modules=['demo', 'test', 'reconstruction', 'analysis', 'check', 'video_preprocessor'],
)
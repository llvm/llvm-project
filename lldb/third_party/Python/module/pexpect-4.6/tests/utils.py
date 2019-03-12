import os

def no_coverage_env():
    "Return a copy of os.environ that won't trigger coverage measurement."
    env = os.environ.copy()
    env.pop('COV_CORE_SOURCE', None)
    return env
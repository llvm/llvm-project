"""Script to exit 0 on the first run, and non-0 on subsequent
runs. This demonstrates a flaky interestingness test.
"""
import sys
import pathlib

# This will exit 0 the first time the script is run, and fail the second time
pathlib.Path(sys.argv[1]).touch(exist_ok=False)

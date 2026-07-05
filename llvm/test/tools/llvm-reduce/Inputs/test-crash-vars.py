import os
import sys

disable_crash_report = os.getenv("LLVM_DISABLE_CRASH_REPORT")
disable_symbolization = os.getenv("LLVM_DISABLE_SYMBOLIZATION")

# Test that this is an explicitly set true value. If we preserve the
# debug environment a pre-set explicit 0 should work.
sys.exit(disable_crash_report != "1" or disable_symbolization != "1")

@echo off

@rem We need to set -u to treat stdin as binary. Python 3 has support for doing
@rem this in code, but I haven't found a way to do this in 2.6 yet.

%PYTHON_EXECUTABLE% -u %LLVM_SRC_ROOT%\test\Scripts\macho-dump %1 %2 %3 %4 %5 %6 %7 %8 %9


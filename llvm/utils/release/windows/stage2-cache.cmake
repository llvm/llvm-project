# Static settings for the stage 2 self-hosted compiler and the packages
# produced from it. The driver passes this file to CMake after the generated
# cache, so the assignments here override the shared release flags.
# Machine-specific paths and PGO flags stay in the PowerShell driver.

set(LLVM_ENABLE_PROJECTS "clang;clang-tools-extra;lld;lldb;flang;mlir" CACHE STRING "" FORCE)
set(LLVM_ENABLE_RUNTIMES "compiler-rt;openmp" CACHE STRING "" FORCE)
set(LLDB_RELOCATABLE_PYTHON "1" CACHE STRING "" FORCE)
set(LLDB_EMBED_PYTHON_HOME "OFF" CACHE STRING "" FORCE)

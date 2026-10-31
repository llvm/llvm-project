# Static settings for the stage 1 bootstrap compiler.
# The driver passes this file to CMake after the generated cache, so the
# assignments here override the shared release flags.

# Stage 1 only needs to compile for the host. Its runtimes are installed
# into a per-target directory so stage 2 can reuse them.
set(LLVM_TARGETS_TO_BUILD "Native" CACHE STRING "" FORCE)
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR "ON" CACHE STRING "" FORCE)

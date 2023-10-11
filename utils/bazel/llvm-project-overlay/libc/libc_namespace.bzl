load("//:vars.bzl", "LLVM_VERSION_MAJOR", "LLVM_VERSION_MINOR", "LLVM_VERSION_PATCH")

# The default libc namespace that encloses all functions.
LIBC_NAMESPACE = "__llvm_libc_{}_{}_{}_git".format(LLVM_VERSION_MAJOR, LLVM_VERSION_MINOR, LLVM_VERSION_PATCH)

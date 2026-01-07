set(LLVM_LINK_COMPONENTS
  support
)

add_llvm_utility(count
  count.c

  DISABLE_PCH_REUSE # no C++ files, prevent CMake error.
  )

set(LLVM_LINK_COMPONENTS
  support
)

add_llvm_utility(count
  count.c

  # No C++ files, prevent CMake error.
  DISABLE_PCH_REUSE
  )

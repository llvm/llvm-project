set(LLVM_LINK_COMPONENTS
  Core
  )

add_llvm_tool(inline1
  t1.c
  t2.c
  t3.c

  DEPENDS
  intrinsics_gen
  )

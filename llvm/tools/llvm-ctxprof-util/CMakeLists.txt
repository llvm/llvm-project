set(LLVM_LINK_COMPONENTS
  Core
  Object
  ProfileData
  Support
  )

add_llvm_tool(llvm-ctxprof-util
llvm-ctxprof-util.cpp

  DEPENDS
  intrinsics_gen
  )

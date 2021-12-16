set(LLVM_LINK_COMPONENTS
  Core
  ProfileData
  Object
  DebugInfoDWARF
  Support
  )

add_llvm_tool(llvm-profdata
  llvm-profdata.cpp

  DEPENDS
  intrinsics_gen
  )

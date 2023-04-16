set(LLVM_LINK_COMPONENTS
  Core
  DebugInfoDWARF
  Object
  ProfileData
  Support
  )

add_llvm_tool(llvm-profdata
  llvm-profdata.cpp

  DEPENDS
  intrinsics_gen
  GENERATE_DRIVER
  )

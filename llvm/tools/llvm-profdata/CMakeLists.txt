set(LLVM_LINK_COMPONENTS
  Core
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

if(LLVM_TOOL_LLVM_DRIVER_BUILD)
  set_property(GLOBAL APPEND PROPERTY LLVM_DRIVER_HIDDEN_TOOL_ALIASES_llvm-profdata gcov-tool)
endif()

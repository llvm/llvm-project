set(LLVM_LINK_COMPONENTS
  Core
  Object
  ProfileData
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(ProfdataOptsTableGen)

add_llvm_tool(llvm-profdata
  llvm-profdata.cpp

  DEPENDS
  intrinsics_gen
  ProfdataOptsTableGen
  GENERATE_DRIVER
  )

if(NOT LLVM_TOOL_LLVM_DRIVER_BUILD)
  target_link_libraries(llvm-profdata PRIVATE LLVMDebuginfod)
endif()

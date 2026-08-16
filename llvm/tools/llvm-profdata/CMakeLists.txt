set(LLVM_LINK_COMPONENTS
  Core
  Object
  Option
  ProfileData
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(ProfDataOptsTableGen)

add_llvm_tool(llvm-profdata
  llvm-profdata.cpp

  DEPENDS
  intrinsics_gen
  ProfDataOptsTableGen
  )

target_link_libraries(llvm-profdata PRIVATE LLVMDebuginfod LLVMHTTP)

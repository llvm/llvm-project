set(LLVM_LINK_COMPONENTS
  CodeGen
  CodeGenData
  Core
  Object
  Support
  )

add_llvm_tool(llvm-cgdata
  llvm-cgdata.cpp

  DEPENDS
  intrinsics_gen
  GENERATE_DRIVER
  )

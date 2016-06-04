set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  DebugInfoPDB
  Object
  Support
  )

add_llvm_tool(llvm-pdbdump
  llvm-pdbdump.cpp
  BuiltinDumper.cpp
  ClassDefinitionDumper.cpp
  CompilandDumper.cpp
  EnumDumper.cpp
  ExternalSymbolDumper.cpp
  FunctionDumper.cpp
  LinePrinter.cpp
  LLVMOutputStyle.cpp
  TypeDumper.cpp
  TypedefDumper.cpp
  VariableDumper.cpp
  )

if(LLVM_USE_SANITIZE_COVERAGE)
  add_subdirectory(fuzzer)
endif()

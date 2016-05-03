set(LLVM_LINK_COMPONENTS
  Object
  Support
  )

add_llvm_tool(llvm-readobj
  ARMAttributeParser.cpp
  ARMWinEHPrinter.cpp
  COFFDumper.cpp
  COFFImportDumper.cpp
  ELFDumper.cpp
  Error.cpp
  llvm-readobj.cpp
  MachODumper.cpp
  ObjDumper.cpp
  Win64EHDumper.cpp
  )

set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  DebugInfoDWARF
  MC
  Object
  Support
  Target
  )

add_llvm_tool(llvm-dsymutil
  dsymutil.cpp
  BinaryHolder.cpp
  CFBundle.cpp
  DebugMap.cpp
  DwarfLinker.cpp
  MachODebugMapParser.cpp
  MachOUtils.cpp

  DEPENDS
  intrinsics_gen
  )

IF(APPLE)
  target_link_libraries(llvm-dsymutil PRIVATE "-framework CoreFoundation")
ENDIF(APPLE)

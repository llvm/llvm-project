set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  DebugInfoDWARF
  MC
  Object
  Support
  Target
  )

add_llvm_tool(dsymutil
  dsymutil.cpp
  BinaryHolder.cpp
  CFBundle.cpp
  DebugMap.cpp
  DwarfLinker.cpp
  MachODebugMapParser.cpp
  MachOUtils.cpp
  NonRelocatableStringpool.cpp

  DEPENDS
  intrinsics_gen
  )

if(APPLE)
  target_link_libraries(dsymutil PRIVATE "-framework CoreFoundation")
endif(APPLE)

if(LLVM_INSTALL_CCTOOLS_SYMLINKS)
  add_llvm_tool_symlink(llvm-dsymutil dsymutil)
endif()


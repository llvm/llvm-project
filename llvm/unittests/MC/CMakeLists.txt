foreach(t ${LLVM_TARGETS_TO_BUILD})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${t})
    add_subdirectory(${t})
  endif()
endforeach()

set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  MC
  MCDisassembler
  Object
  Support
  )

add_llvm_unittest(MCTests
  Disassembler.cpp
  DwarfLineTables.cpp
  DwarfLineTableHeaders.cpp
  MCInstPrinter.cpp
  StringTableBuilderTest.cpp
  TargetRegistry.cpp
  MCDisassemblerTest.cpp
  )


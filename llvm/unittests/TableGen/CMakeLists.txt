set(LLVM_LINK_COMPONENTS
  TableGen
  Support
  )

set(LLVM_TARGET_DEFINITIONS Automata.td)

tablegen(LLVM AutomataTables.inc -gen-searchable-tables)
tablegen(LLVM AutomataAutomata.inc -gen-automata)
add_public_tablegen_target(AutomataTestTableGen)

add_llvm_unittest(TableGenTests DISABLE_LLVM_LINK_LLVM_DYLIB
  AutomataTest.cpp
  CodeExpanderTest.cpp
  ParserEntryPointTest.cpp
  )

target_link_libraries(TableGenTests PRIVATE LLVMTableGenCommon LLVMTableGen)

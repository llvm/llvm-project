set(LLVM_LINK_COMPONENTS
  TableGen
  Support
  )

set(LLVM_TARGET_DEFINITIONS Automata.td)

llvm_tablegen(AutomataTables.inc -gen-searchable-tables)
llvm_tablegen(AutomataAutomata.inc -gen-automata)
add_public_tablegen_target(AutomataTestTableGen)

add_llvm_unittest(TableGenTests
  AutomataTest.cpp
  CodeExpanderTest.cpp
  ParserEntryPointTest.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB
  )

target_link_libraries(TableGenTests PRIVATE LLVMTableGenCommon LLVMTableGen)

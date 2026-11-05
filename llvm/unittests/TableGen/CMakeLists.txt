set(LLVM_LINK_COMPONENTS
  TableGen
  Support
  )

set(LLVM_TARGET_DEFINITIONS Automata.td)

llvm_tablegen(AutomataTables.inc -gen-searchable-tables)
llvm_tablegen(AutomataAutomata.inc -gen-automata)

set(LLVM_TARGET_DEFINITIONS
  ${LLVM_MAIN_SRC_DIR}/test/TableGen/IntrinsicFeatureChecksSingleGuard.td)
llvm_tablegen(IntrinsicFeatureChecksSingleGuard.inc
  -gen-intrinsic-feature-checks
  -intrinsic-feature-prefix=test
  -intrinsic-feature-namespace=TestTarget
  DEPENDS
  ${LLVM_MAIN_SRC_DIR}/include/llvm/IR/Intrinsics.td
  ${LLVM_MAIN_SRC_DIR}/include/llvm/IR/IntrinsicsAArch64.td
  ${LLVM_MAIN_SRC_DIR}/include/llvm/Target/TargetSubtarget.td)
add_public_tablegen_target(AutomataTestTableGen)

add_llvm_unittest(TableGenTests
  AutomataTest.cpp
  CodeExpanderTest.cpp
  FeatureGuardParserTest.cpp
  IntrinsicFeatureEmitterTest.cpp
  ParserEntryPointTest.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB
  )

target_link_libraries(TableGenTests PRIVATE LLVMTableGenCommon LLVMTableGen)

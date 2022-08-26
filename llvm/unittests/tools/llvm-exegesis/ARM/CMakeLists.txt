add_llvm_exegesis_unittest_includes(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/ARM
  ${LLVM_LIBRARY_DIR}/Target/ARM
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

add_llvm_exegesis_unittest_link_components(
  Core
  MC
  MCParser
  Object
  Support
  Symbolize
  ARM
  )

add_llvm_exegesis_unittest_sources(
  AssemblerTest.cpp
  )

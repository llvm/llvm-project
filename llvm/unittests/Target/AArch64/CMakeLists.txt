include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64
  ${LLVM_BINARY_DIR}/lib/Target/AArch64
  )

set(LLVM_LINK_COMPONENTS
  AArch64CodeGen
  AArch64Utils
  AArch64Desc
  AArch64Info
  AsmParser
  CodeGen
  Core
  GlobalISel
  MC
  MIRParser
  SelectionDAG
  Support
  Target
  )

add_llvm_target_unittest(AArch64Tests
  InstSizes.cpp
  DecomposeStackOffsetTest.cpp
  MatrixRegisterAliasing.cpp
  SMEAttributesTest.cpp
  )

set_property(TARGET AArch64Tests PROPERTY FOLDER "Tests/UnitTests/TargetTests")

include_directories(
  ${PROJECT_SOURCE_DIR}/lib/Target/AMDGPU
  ${PROJECT_BINARY_DIR}/lib/Target/AMDGPU
  )

set(LLVM_LINK_COMPONENTS
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  MC
  MIRParser
  Support
  TargetParser
  )

add_llvm_target_unittest(AMDGPUTests
  AMDGPUUnitTests.cpp
  CSETest.cpp
  DwarfRegMappings.cpp
  ExecMayBeModifiedBeforeAnyUse.cpp
  LiveRegUnits.cpp
  NextUseAnalysisTest.cpp
  PALMetadata.cpp
  )

# Define test directory for NextUseAnalysis tests
target_compile_definitions(AMDGPUTests PRIVATE
  AMDGPU_NUA_TEST_DIR="${LLVM_MAIN_SRC_DIR}/test/CodeGen/AMDGPU/NextUseAnalysis")

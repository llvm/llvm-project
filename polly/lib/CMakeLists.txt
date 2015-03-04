set(LLVM_NO_RTTI 1)

if (PLUTO_FOUND)
  set(POLLY_PLUTO_FILES Transform/Pluto.cpp)
endif (PLUTO_FOUND)

set(POLLY_JSON_FILES
    JSON/json_reader.cpp
    JSON/json_value.cpp
    JSON/json_writer.cpp
)

set(ISL_CODEGEN_FILES
    CodeGen/IslAst.cpp
    CodeGen/IslExprBuilder.cpp
    CodeGen/IslCodeGeneration.cpp)

if (GPU_CODEGEN)
  set (GPGPU_CODEGEN_FILES
       CodeGen/PTXGenerator.cpp)
endif (GPU_CODEGEN)

add_polly_library(Polly
  Analysis/Dependences.cpp
  Analysis/ScopDetection.cpp
  Analysis/ScopDetectionDiagnostic.cpp
  Analysis/ScopInfo.cpp
  Analysis/ScopGraphPrinter.cpp
  Analysis/ScopPass.cpp
  Analysis/TempScopInfo.cpp
  CodeGen/BlockGenerators.cpp
  ${ISL_CODEGEN_FILES}
  CodeGen/LoopGenerators.cpp
  CodeGen/IRBuilder.cpp
  CodeGen/Utils.cpp
  CodeGen/RuntimeDebugBuilder.cpp
  ${GPGPU_CODEGEN_FILES}
  Exchange/JSONExporter.cpp
  Support/GICHelper.cpp
  Support/SCEVValidator.cpp
  Support/RegisterPasses.cpp
  Support/ScopHelper.cpp
  ${POLLY_JSON_FILES}
  Transform/Canonicalization.cpp
  Transform/CodePreparation.cpp
  Transform/DeadCodeElimination.cpp
  Transform/IndependentBlocks.cpp
  Transform/ScheduleOptimizer.cpp
  ${POLLY_PLUTO_FILES}
  )

if (BUILD_SHARED_LIBS)
  target_link_libraries(Polly
    LLVMSupport
    LLVMCore
    LLVMScalarOpts
    LLVMInstCombine
    LLVMTransformUtils
    LLVMAnalysis
    LLVMipo
    LLVMProfileData
  )
  link_directories(
      ${LLVM_LIBRARY_DIR}
  )
endif()

# Build a monolithic Polly.a and a thin module LLVMPolly.moduleext that links to
# that static library.
add_polly_loadable_module(LLVMPolly
  Polly.cpp
)

if (TARGET intrinsics_gen)
  # Check if we are building as part of an LLVM build
  add_dependencies(Polly intrinsics_gen)
endif()

target_link_libraries(LLVMPolly Polly)

set_target_properties(LLVMPolly
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")

set(LLVM_NO_RTTI 1)

set(POLLY_JSON_FILES
    JSON/json_reader.cpp
    JSON/json_value.cpp
    JSON/json_writer.cpp
)

set(ISL_CODEGEN_FILES
    CodeGen/IslAst.cpp
    CodeGen/IslExprBuilder.cpp
    CodeGen/IslNodeBuilder.cpp
    CodeGen/CodeGeneration.cpp)

if (GPU_CODEGEN)
  set (GPGPU_CODEGEN_FILES
       CodeGen/PPCGCodeGeneration.cpp
       )
endif (GPU_CODEGEN)

# Compile ISL into a separate library.
add_subdirectory(External)

set(POLLY_HEADER_FILES)
if (MSVC_IDE OR XCODE)
  file(GLOB_RECURSE POLLY_HEADER_FILES "${POLLY_SOURCE_DIR}/include/polly/*.h")
endif ()

add_polly_library(Polly
  Analysis/DependenceInfo.cpp
  Analysis/PolyhedralInfo.cpp
  Analysis/ScopDetection.cpp
  Analysis/ScopDetectionDiagnostic.cpp
  Analysis/ScopInfo.cpp
  Analysis/ScopBuilder.cpp
  Analysis/ScopGraphPrinter.cpp
  Analysis/ScopPass.cpp
  CodeGen/BlockGenerators.cpp
  ${ISL_CODEGEN_FILES}
  CodeGen/LoopGenerators.cpp
  CodeGen/IRBuilder.cpp
  CodeGen/Utils.cpp
  CodeGen/RuntimeDebugBuilder.cpp
  CodeGen/CodegenCleanup.cpp
  ${GPGPU_CODEGEN_FILES}
  Exchange/JSONExporter.cpp
  Support/GICHelper.cpp
  Support/SCEVAffinator.cpp
  Support/SCEVValidator.cpp
  Support/RegisterPasses.cpp
  Support/ScopHelper.cpp
  Support/ScopLocation.cpp
  ${POLLY_JSON_FILES}
  Transform/Canonicalization.cpp
  Transform/CodePreparation.cpp
  Transform/DeadCodeElimination.cpp
  Transform/ScheduleOptimizer.cpp
  Transform/FlattenSchedule.cpp
  Transform/FlattenAlgo.cpp
  ${POLLY_HEADER_FILES}
  )

if (GPU_CODEGEN)
  target_link_libraries(Polly PollyPPCG)
endif (GPU_CODEGEN)

target_link_libraries(Polly PollyISL)

if (BUILD_SHARED_LIBS)
  target_link_libraries(Polly
    LLVMSupport
    LLVMCore
    LLVMScalarOpts
    LLVMInstCombine
    LLVMTransformUtils
    LLVMAnalysis
    LLVMipo
    LLVMMC
# The libraries below are required for darwin: http://PR26392
    LLVMBitReader
    LLVMMCParser
    LLVMObject
    LLVMProfileData
    LLVMTarget
    LLVMVectorize
  )
  link_directories(
    ${LLVM_LIBRARY_DIR}
  )
elseif (LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(Polly
    LLVM
  )
  link_directories(
    ${LLVM_LIBRARY_DIR}
  )
endif()

# Build a monolithic Polly.a and a thin module LLVMPolly.moduleext that links to
# that static library.
if (MSVC)
  # Add dummy target, because loadable modules are not supported on Windows
  add_custom_target(LLVMPolly)
  set_target_properties(LLVMPolly PROPERTIES FOLDER "Polly")
else ()
  add_polly_loadable_module(LLVMPolly
    Polly.cpp
  )

  target_link_libraries(LLVMPolly Polly)

  set_target_properties(LLVMPolly
    PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX "")
endif ()

if (TARGET intrinsics_gen)
  # Check if we are building as part of an LLVM build
  add_dependencies(Polly intrinsics_gen)
endif()


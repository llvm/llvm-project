set(LLVM_NO_RTTI 1)

set(POLLY_JSON_FILES
    External/JSON/json_reader.cpp
    External/JSON/json_value.cpp
    External/JSON/json_writer.cpp
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

# Use an object-library to add the same files to multiple libs without requiring
# the sources them to be recompiled for each of them.
add_library(PollyCore OBJECT
  Analysis/DependenceInfo.cpp
  Analysis/PolyhedralInfo.cpp
  Analysis/ScopDetection.cpp
  Analysis/ScopDetectionDiagnostic.cpp
  Analysis/ScopInfo.cpp
  Analysis/ScopBuilder.cpp
  Analysis/ScopGraphPrinter.cpp
  Analysis/ScopPass.cpp
  Analysis/PruneUnprofitable.cpp
  CodeGen/BlockGenerators.cpp
  ${ISL_CODEGEN_FILES}
  CodeGen/LoopGenerators.cpp
  CodeGen/IRBuilder.cpp
  CodeGen/Utils.cpp
  CodeGen/RuntimeDebugBuilder.cpp
  CodeGen/CodegenCleanup.cpp
  CodeGen/PerfMonitor.cpp
  ${GPGPU_CODEGEN_FILES}
  Exchange/JSONExporter.cpp
  Support/GICHelper.cpp
  Support/SCEVAffinator.cpp
  Support/SCEVValidator.cpp
  Support/RegisterPasses.cpp
  Support/ScopHelper.cpp
  Support/ScopLocation.cpp
  Support/ISLTools.cpp
  Support/DumpModulePass.cpp
  ${POLLY_JSON_FILES}
  Transform/Canonicalization.cpp
  Transform/CodePreparation.cpp
  Transform/DeadCodeElimination.cpp
  Transform/ScheduleOptimizer.cpp
  Transform/FlattenSchedule.cpp
  Transform/FlattenAlgo.cpp
  Transform/DeLICM.cpp
  Transform/Simplify.cpp
  ${POLLY_HEADER_FILES}
  )
set_target_properties(PollyCore PROPERTIES FOLDER "Polly")

# Create the library that can be linked into LLVM's tools and Polly's unittests.
# It depends on all library it needs, such that with
# LLVM_POLLY_LINK_INTO_TOOLS=ON, its dependencies like PollyISL are linked as
# well.
add_polly_library(Polly $<TARGET_OBJECTS:PollyCore>)
target_link_libraries(Polly
  ${ISL_TARGET}
)

# Additional dependencies for Polly-ACC.
if (GPU_CODEGEN)
  target_link_libraries(Polly PollyPPCG)
endif ()

# Add Polly's LLVM dependencies.
# When built inside the LLVM source tree, these are CMake targets that will
# depend on their dependencies and CMake ensures they are added in the right
# order.
# If Polly is built independently, just add all LLVM libraries. LLVM_ROOT_DIR
# might have been configured to compile to individual libraries or a single
# libLLVM.so (called dylib), reported by llvm-config, so there is no fixed list
# of required libraries.
if (DEFINED LLVM_MAIN_SRC_DIR)

  # Polly-ACC requires the NVPTX backend to work. Ask LLVM about its libraries.
  set(nvptx_libs)
  if (GPU_CODEGEN)
    # This call emits an error if they NVPTX backend is not enable.
    llvm_map_components_to_libnames(nvptx_libs NVPTX)
  endif ()

  if (LLVM_LINK_LLVM_DYLIB)
    # The shlib/dylib contains all the LLVM components
    # (including NVPTX is enabled) already. Adding them to target_link_libraries
    # would cause them being twice in the address space
    # (their LLVM*.a/so and their copies in libLLVM.so)
    # which results in errors when the two instances try to register the same
    # command-line switches.
    target_link_libraries(Polly LLVM)
  else ()
    target_link_libraries(Polly
      LLVMSupport
      LLVMCore
      LLVMScalarOpts
      LLVMInstCombine
      LLVMTransformUtils
      LLVMAnalysis
      LLVMipo
      LLVMMC
      ${nvptx_libs}
# The libraries below are required for darwin: http://PR26392
      LLVMBitReader
      LLVMMCParser
      LLVMObject
      LLVMProfileData
      LLVMTarget
      LLVMVectorize
    )
  endif ()
else ()
  # When Polly-ACC is enabled, we assume that the host LLVM was also built with
  # the NVPTX target enabled.
  target_link_libraries(Polly
    ${LLVM_LIBS}
    ${LLVM_SYSTEM_LIBS}
  )
endif ()

# Create a loadable module Polly.so that can be loaded using
# LLVM's/clang's "-load" option.
if (MSVC)
  # Add dummy target, because loadable modules are not supported on Windows
  add_custom_target(LLVMPolly)
  set_target_properties(LLVMPolly PROPERTIES FOLDER "Polly")
else ()
  add_polly_loadable_module(LLVMPolly
    Polly.cpp
    $<TARGET_OBJECTS:PollyCore>
  )

  # Only add the dependencies that are not part of LLVM. The latter are assumed
  # to be already available in the address space the module is loaded into.
  # Adding them once more would have the effect that both copies try to register
  # the same command line options, to which LLVM reacts with an error.
  # If Polly-ACC is enabled, the NVPTX target is also expected to reside in the
  # hosts. This is not the case for bugpoint. Use LLVM_POLLY_LINK_INTO_TOOLS=ON
  # instead which will automatically resolve the additional dependencies by
  # Polly.
  target_link_libraries(LLVMPolly ${ISL_TARGET})
  if (GPU_CODEGEN)
    target_link_libraries(LLVMPolly PollyPPCG)
  endif ()

  set_target_properties(LLVMPolly
    PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX "")
endif ()

if (TARGET intrinsics_gen)
  # Check if we are building as part of an LLVM build
  add_dependencies(PollyCore intrinsics_gen)
endif()


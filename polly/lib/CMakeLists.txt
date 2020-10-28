set(LLVM_NO_RTTI 1)

set(ISL_CODEGEN_FILES
    CodeGen/IslAst.cpp
    CodeGen/IslExprBuilder.cpp
    CodeGen/IslNodeBuilder.cpp
    CodeGen/CodeGeneration.cpp)

if (GPU_CODEGEN)
  set (GPGPU_CODEGEN_FILES
       CodeGen/PPCGCodeGeneration.cpp
       CodeGen/ManagedMemoryRewrite.cpp
       )
endif (GPU_CODEGEN)

# Compile ISL into a separate library.
add_subdirectory(External)

set(POLLY_HEADER_FILES)
if (MSVC_IDE OR XCODE)
  file(GLOB_RECURSE POLLY_HEADER_FILES "${POLLY_SOURCE_DIR}/include/polly/*.h")
endif ()

set(POLLY_COMPONENTS
    Support
    Core
    ScalarOpts
    InstCombine
    TransformUtils
    Analysis
    ipo
    MC
    Passes
    Linker
    IRReader
    Analysis
    # The libraries below are required for darwin: http://PR26392
    BitReader
    MCParser
    Object
    ProfileData
    Target
    Vectorize
)

# Polly-ACC requires the NVPTX backend to work. Ask LLVM about its libraries.
if (GPU_CODEGEN)
  # This call emits an error if they NVPTX backend is not enable.
  list(APPEND POLLY_COMPONENTS NVPTX)
endif ()

# Use an object-library to add the same files to multiple libs without requiring
# the sources them to be recompiled for each of them.
add_llvm_pass_plugin(Polly
  NO_MODULE
  SUBPROJECT Polly
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
  CodeGen/LoopGeneratorsGOMP.cpp
  CodeGen/LoopGeneratorsKMP.cpp
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
  Support/VirtualInstruction.cpp
  Transform/Canonicalization.cpp
  Transform/CodePreparation.cpp
  Transform/DeadCodeElimination.cpp
  Transform/ScheduleOptimizer.cpp
  Transform/ScheduleTreeTransform.cpp
  Transform/FlattenSchedule.cpp
  Transform/FlattenAlgo.cpp
  Transform/ForwardOpTree.cpp
  Transform/DeLICM.cpp
  Transform/ZoneAlgo.cpp
  Transform/Simplify.cpp
  Transform/MaximalStaticExpansion.cpp
  Transform/RewriteByReferenceParameters.cpp
  Transform/ScopInliner.cpp
  ${POLLY_HEADER_FILES}

  LINK_COMPONENTS
  ${POLLY_COMPONENTS}
  )
set_target_properties(obj.Polly PROPERTIES FOLDER "Polly")
set_target_properties(Polly PROPERTIES FOLDER "Polly")

if (MSVC_IDE OR XCODE)
  # Configure source groups for Polly source files. By default, in the IDE there
  # will be a source and include folder. In the source folder will be all the
  # source files in a flat list, and in the include folder will be all the
  # headers in a flat list. Sets the CMake source_group for each folder such
  # the organization of the sources and headers in the IDE matches how it is
  # laid out on disk
  setup_polly_source_groups(${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../include/polly)
endif()

# Create the library that can be linked into LLVM's tools and Polly's unittests.
# It depends on all library it needs, such that with
# LLVM_POLLY_LINK_INTO_TOOLS=ON, its dependencies like PollyISL are linked as
# well.
target_link_libraries(Polly PUBLIC
  ${ISL_TARGET}
)

# Additional dependencies for Polly-ACC.
if (GPU_CODEGEN)
  target_link_libraries(Polly PUBLIC PollyPPCG)
endif ()

if (NOT LLVM_LINK_LLVM_DYLIB AND NOT LLVM_POLLY_LINK_INTO_TOOLS)
    # Polly-ACC requires the NVPTX target to be present in the executable it is linked to
    set_property(TARGET bugpoint APPEND PROPERTY LINK_LIBRARIES LLVMTarget)
endif ()

# Create a loadable module Polly.so that can be loaded using
# LLVM's/clang's "-load" option.
if (MSVC OR NOT LLVM_ENABLE_PIC)
  # Add dummy target, either because loadable modules are not supported
  # as on Windows or because PIC code has been disabled
  add_custom_target(LLVMPolly)
  set_target_properties(LLVMPolly PROPERTIES FOLDER "Polly")
else ()
  add_polly_loadable_module(LLVMPolly
    Plugin/Polly.cpp
    $<TARGET_OBJECTS:obj.Polly>
  )

  # Only add the dependencies that are not part of LLVM. The latter are assumed
  # to be already available in the address space the module is loaded into.
  # Adding them once more would have the effect that both copies try to register
  # the same command line options, to which LLVM reacts with an error.
  # If Polly-ACC is enabled, the NVPTX target is also expected to reside in the
  # hosts. This is not the case for bugpoint. Use LLVM_POLLY_LINK_INTO_TOOLS=ON
  # instead which will automatically resolve the additional dependencies by
  # Polly.
  target_link_libraries(LLVMPolly PUBLIC ${ISL_TARGET})
  if (GPU_CODEGEN)
    target_link_libraries(LLVMPolly PUBLIC PollyPPCG)
  endif ()

  set_target_properties(LLVMPolly
    PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX "")
endif ()

if (TARGET intrinsics_gen)
  # Check if we are building as part of an LLVM build
  add_dependencies(obj.Polly intrinsics_gen)
endif()

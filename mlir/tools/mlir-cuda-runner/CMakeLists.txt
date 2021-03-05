set(LLVM_OPTIONAL_SOURCES
  mlir-cuda-runner.cpp
  )
set(LLVM_LINK_COMPONENTS
  Core
  Support
)

if(MLIR_CUDA_RUNNER_ENABLED)
  if (NOT ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR
      "Building the mlir cuda runner requires the NVPTX backend")
  endif()

  # Configure CUDA runner support. Using check_language first allows us to give
  # a custom error message.
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "Building the mlir cuda runner requires a working CUDA install")
  endif()

  # We need the libcuda.so library.
  find_library(CUDA_RUNTIME_LIBRARY cuda)

  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  set(LIBS
    ${conversion_libs}
    MLIRJitRunner
    MLIRAnalysis
    MLIRAsync
    MLIREDSC
    MLIRExecutionEngine
    MLIRGPU
    MLIRIR
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    MLIRNVVMIR
    MLIRParser
    MLIRStandard
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRNVVMToLLVMIRTranslation
    MLIRTransforms
    MLIRTranslation
    ${CUDA_RUNTIME_LIBRARY}
  )

  # Manually expand the target library, since our MLIR libraries
  # aren't plugged into the LLVM dependency tracking. If we don't
  # do this then we can't insert the CodeGen library after ourselves
  llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens)
  # Prepend LLVM in front of every target, this is how the library
  # are named with CMake
  SET(targets_to_link)
  FOREACH(t ${TARGET_LIBS})
    LIST(APPEND targets_to_link "LLVM${t}")
  ENDFOREACH(t)

  add_llvm_tool(mlir-cuda-runner
    mlir-cuda-runner.cpp

    DEPENDS
    mlir_cuda_runtime
    )
  target_include_directories(mlir-cuda-runner
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  llvm_update_compile_flags(mlir-cuda-runner)
  target_link_libraries(mlir-cuda-runner PRIVATE ${LIBS} ${targets_to_link})

endif()

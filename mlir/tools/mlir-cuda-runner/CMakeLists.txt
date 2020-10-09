set(LLVM_OPTIONAL_SOURCES
  cuda-runtime-wrappers.cpp
  mlir-cuda-runner.cpp
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

  add_llvm_library(cuda-runtime-wrappers SHARED
    cuda-runtime-wrappers.cpp
  )
  target_include_directories(cuda-runtime-wrappers
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    LLVMSupport
  )
  target_link_libraries(cuda-runtime-wrappers
    PUBLIC
    LLVMSupport
    ${CUDA_RUNTIME_LIBRARY}
  )

  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRJitRunner
    MLIRAnalysis
    MLIREDSC
    MLIRExecutionEngine
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRTargetLLVMIR
    MLIRTargetNVVMIR
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
    cuda-runtime-wrappers

    LINK_COMPONENTS
    Core
    Support
    )
  target_include_directories(mlir-cuda-runner
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  llvm_update_compile_flags(mlir-cuda-runner)
  target_link_libraries(mlir-cuda-runner PRIVATE ${LIBS} ${targets_to_link})

endif()

if (MLIR_ENABLE_CUDA_RUNNER)
  # Configure CUDA support. Using check_language first allows us to give a
  # custom error message.
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "Building the mlir cuda runner requires a working CUDA install")
  endif()

  # We need the libcuda.so library.
  find_library(CUDA_RUNTIME_LIBRARY cuda HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} REQUIRED)

  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  set(LIBS
    ${conversion_libs}

    MLIRAnalysis
    MLIRArithDialect
    MLIRBuiltinToLLVMIRTranslation
    MLIRExecutionEngine
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRIR
    MLIRJitRunner
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRToLLVMIRTranslationRegistration
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRParser
    MLIRSPIRVDialect
    MLIRSPIRVTransforms
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIRTranslateLib
    MLIRVectorDialect
    MLIRVectorToLLVM
  )

  # Exclude tests from libMLIR.so
  add_mlir_library(MLIRNVVMTestPasses
    TestLowerToNVVM.cpp

    EXCLUDE_FROM_LIBMLIR

    LINK_LIBS PUBLIC
    ${LIBS}
    )
endif()

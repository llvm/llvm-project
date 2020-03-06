set(LLVM_OPTIONAL_SOURCES
  FileUtilities.cpp
  JitRunner.cpp
  MlirOptMain.cpp
  StorageUniquer.cpp
  ToolUtilities.cpp
  TranslateClParser.cpp
)

add_mlir_library(MLIRSupport
  FileUtilities.cpp
  StorageUniquer.cpp
  ToolUtilities.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support
  )
target_link_libraries(MLIRSupport
  PUBLIC
  LLVMSupport
  ${LLVM_PTHREAD_LIB})

add_mlir_library(MLIROptLib
  MlirOptMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support
  )
target_link_libraries(MLIROptLib
  PUBLIC
  MLIRPass
  MLIRParser
  LLVMSupport
  MLIRSupport
  )

add_mlir_library(MLIRTranslateClParser
  TranslateClParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support
  )
target_link_libraries(MLIRTranslateClParser
  PUBLIC
  LLVMSupport
  MLIRIR
  MLIRTranslation
  MLIRParser)

add_llvm_library(MLIRJitRunner
  JitRunner.cpp
)
target_link_libraries(MLIRJitRunner
  PUBLIC
  MLIRExecutionEngine
  MLIRIR
  MLIRParser
  MLIRStandardOps
  MLIRTargetLLVMIR
  MLIRTransforms
  MLIRStandardToLLVM
  MLIRSupport
  LLVMCore
  LLVMSupport
)

set(LLVM_OPTIONAL_SOURCES
  InferIntRangeCommon.cpp
  MemorySlotUtils.cpp
  )

add_mlir_library(MLIRInferIntRangeCommon
    InferIntRangeCommon.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces/Utils

    DEPENDS
    MLIRInferIntRangeInterfaceIncGen

    LINK_LIBS PUBLIC
    MLIRShapedOpInterfaces
    MLIRInferIntRangeInterface
    MLIRIR
)

add_mlir_library(MLIRMemorySlotUtils
    MemorySlotUtils.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces/Utils

    LINK_LIBS PUBLIC
    MLIRIR
)

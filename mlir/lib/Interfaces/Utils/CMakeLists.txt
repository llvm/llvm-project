add_mlir_library(MLIRInferIntRangeCommon
    InferIntRangeCommon.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces/Utils

    DEPENDS
    MLIRInferIntRangeInterfaceIncGen

    LINK_LIBS PUBLIC
    MLIRInferIntRangeInterface
    MLIRIR
)

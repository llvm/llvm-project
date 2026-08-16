add_mlir_dialect_library(MLIRTileReducer
        TileReducerDialect.cpp
        TileReducerOps.cpp
        TileReducerTypes.cpp
        TileReducerAnalyses.cpp
        TileReducerBufferConversion.cpp
        TileReducerPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/TileReducer

        DEPENDS
        MLIRTileReducerOpsIncGen
        MLIRTileReducerPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRAsmParser
        MLIRArithDialect
        MLIRControlFlowInterfaces
        MLIRFuncDialect
        MLIRFuncTransforms
        MLIRMemRefDialect
        MLIRInferTypeOpInterface
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTransformUtils
        )

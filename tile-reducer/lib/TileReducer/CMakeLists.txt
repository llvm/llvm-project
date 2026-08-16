add_mlir_dialect_library(MLIRTileReducer
        TileReducerDialect.cpp
        TileReducerOps.cpp
        TileReducerTypes.cpp
        TileReducerAnalyses.cpp
        TileReducerBufferConversion.cpp
        TileReducerLinalgConversion.cpp
        TileReducerLinalgTiling.cpp
        TileReducerTransformOps.cpp
        TileReducerPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/TileReducer

        DEPENDS
        MLIRTileReducerOpsIncGen
        MLIRTileReducerPassesIncGen
        MLIRTileReducerTransformOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRAsmParser
        MLIRArithDialect
        MLIRControlFlowInterfaces
        MLIRFuncDialect
        MLIRAffineDialect
        MLIRFuncTransforms
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRMemRefDialect
        MLIRSCFDialect
        MLIRSCFTransforms
        MLIRTransformDialect
        MLIRInferTypeOpInterface
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTransformUtils
        )

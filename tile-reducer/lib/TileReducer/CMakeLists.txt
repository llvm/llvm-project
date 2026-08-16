add_mlir_dialect_library(MLIRTileReducer
        TileReducerDialect.cpp
        TileReducerOps.cpp
        TileReducerTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/TileReducer

        DEPENDS
        MLIRTileReducerOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRAsmParser
        )

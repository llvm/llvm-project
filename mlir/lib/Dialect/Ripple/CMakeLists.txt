
add_mlir_dialect_library(MLIRRippleDialect RippleDialect.cpp

                             ADDITIONAL_HEADER_DIRS ${MLIR_MAIN_INCLUDE_DIR} /
                         mlir / Dialect /
                         Ripple

                             DEPENDS MLIRRippleOpsIncGen

                                 LINK_LIBS PUBLIC MLIRDialect MLIRIR
                                     MLIRUBDialect)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_library(TosaFuserRegistration
    Pipelines.cpp

    LINK_LIBS PUBLIC
    ${dialect_libs}
    ${conversion_libs}
    MLIRPass
    MLIRTransforms
    MLIRGPUTransforms
)
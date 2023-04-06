get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        MLIRIR
        MLIRPass
        MLIRStandalone
        MLIRTransformUtils
        )

add_mlir_dialect_library(StandalonePlugin
        SHARED
        standalone-plugin.cpp

        DEPENDS
        MLIRStandalone
        )

llvm_update_compile_flags(StandalonePlugin)
target_link_libraries(StandalonePlugin PRIVATE ${LIBS})

mlir_check_all_link_libraries(StandalonePlugin)

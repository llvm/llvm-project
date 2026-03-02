# Enable errors for any global constructors.
add_flag_if_supported("-Werror=global-constructors" WERROR_GLOBAL_CONSTRUCTOR)

add_subdirectory(Analysis)
add_subdirectory(AsmParser)
add_subdirectory(Bytecode)
add_subdirectory(Conversion)
add_subdirectory(Debug)
add_subdirectory(Dialect)
add_subdirectory(IR)
add_subdirectory(Interfaces)
add_subdirectory(Parser)
add_subdirectory(Pass)
add_subdirectory(Query)
add_subdirectory(Reducer)
add_subdirectory(Remark)
add_subdirectory(Rewrite)
add_subdirectory(Support)
add_subdirectory(TableGen)
add_subdirectory(Target)
add_subdirectory(Tools)
add_subdirectory(Transforms)
add_subdirectory(ExecutionEngine)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_library(MLIRRegisterAllDialects
  RegisterAllDialects.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  ${dialect_libs}
  )

add_mlir_library(MLIRRegisterAllPasses
  RegisterAllPasses.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  ${dialect_libs} # Some passes are part of the dialect libs
  ${conversion_libs}
  )

add_mlir_library(MLIRRegisterAllExtensions
  RegisterAllExtensions.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  )

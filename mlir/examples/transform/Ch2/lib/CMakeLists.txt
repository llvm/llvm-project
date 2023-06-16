add_mlir_library(
  # Library called MyExtension.
  MyExtensionCh2

  # Built from the following source files.
  MyExtension.cpp

  # Do not make this part of the main library distribution.
  # Needed for the example only.
  EXCLUDE_FROM_LIBMLIR

  # Make includes visible without top-level path.
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/examples/transform/Ch2/include

  # Make sure ODS declaration and definitions are generated before compiling this.
  DEPENDS
  MyExtensionCh2IncGen

  # Link in the transform dialect, an all generated dialects.
  LINK_LIBS PUBLIC
  MLIRTransformDialect
  MLIRFuncDialect
  MLIRSCFDialect
)

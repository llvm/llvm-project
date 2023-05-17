add_mlir_library(MLIRPluginsLib
  DialectPlugin.cpp
  PassPlugin.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/Plugins

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  )

add_mlir_dialect_library(MLIRXeGPUuArch
  IntelGpuXe2.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/XeGPU/uArch

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialectUtils
)


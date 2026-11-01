add_mlir_library(MLIRRemarkStreamer
  LLVMRemarkImport.cpp
  RemarkStreamer.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Remark

  LINK_LIBS PUBLIC
    MLIRIR

  LINK_COMPONENTS
    Remarks
    Core
    BitstreamReader
  )

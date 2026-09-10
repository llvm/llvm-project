add_mlir_library(MLIROpenACCMPCommon
AtomicInterfaces.cpp

ADDITIONAL_HEADER_DIRS
${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenACCMPCommon/Interfaces

DEPENDS
MLIRArithOpsIncGen
MLIRAtomicInterfacesIncGen

LINK_LIBS PUBLIC
MLIRArithDialect
MLIRIR
)

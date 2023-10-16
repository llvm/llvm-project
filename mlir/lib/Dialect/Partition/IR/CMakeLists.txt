add_mlir_dialect_library(MLIRPartition
   PartitionDialect.cpp

   ADDITIONAL_HEADER_DIRS
   ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Partition

   DEPENDS
   MLIRPartitionOpsIncGen

   LINK_LIBS PUBLIC
   MLIRDialect
   MLIRIR
   )

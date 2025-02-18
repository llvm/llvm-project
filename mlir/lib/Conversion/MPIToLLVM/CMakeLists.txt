find_path(MPI_C_HEADER_DIR mpi.h
    PATHS $ENV{I_MPI_ROOT}/include
          $ENV{MPI_HOME}/include
          $ENV{MPI_ROOT}/include)

add_mlir_conversion_library(MLIRMPIToLLVM
  MPIToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MPIToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMPIDialect
)

if(MPI_C_HEADER_DIR)
  message(STATUS "found MPI_C_HEADER_DIR: ${MPI_C_HEADER_DIR}")
  target_include_directories(obj.MLIRMPIToLLVM PRIVATE ${MPI_C_HEADER_DIR})
  target_compile_definitions(obj.MLIRMPIToLLVM PUBLIC FOUND_MPI_C_HEADER=1)
else()
  message(WARNING "MPI not found, falling back to definitions from MPICH")
endif()

add_subdirectory(AffineOps)
add_subdirectory(FxpMathOps)
add_subdirectory(GPU)
add_subdirectory(Linalg)
add_subdirectory(LLVMIR)
add_subdirectory(LoopOps)
add_subdirectory(OpenMP)
add_subdirectory(QuantOps)
add_subdirectory(SDBM)
add_subdirectory(Shape)
add_subdirectory(SPIRV)
add_subdirectory(StandardOps)
add_subdirectory(VectorOps)


set(LLVM_OPTIONAL_SOURCES
  Traits.cpp
)

add_llvm_library(MLIRDialect
  Traits.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect
  )
target_link_libraries(MLIRDialect MLIRIR)

# Create a dummy MLIRAllDialects library for the purpose
# of having an easy tracking of all dialects when linking
# them in tools.
# Empty libraries aren't possible with CMake, create a dummy file.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/all_dialects.c "typedef int make_iso_compilers_happy;\n")
add_llvm_library(MLIRAllDialects
  ${CMAKE_CURRENT_BINARY_DIR}/all_dialects.c
)
target_link_libraries(MLIRAllDialects
  MLIRAffineOps
  MLIRFxpMathOps
  MLIRGPU
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRROCDLIR
  MLIRLinalgOps
  MLIRLoopOps
  MLIROpenMP
  MLIRQuantOps
  MLIRSDBM
  MLIRShape
  MLIRSPIRV
  MLIRStandardOps
  MLIRVectorOps
)

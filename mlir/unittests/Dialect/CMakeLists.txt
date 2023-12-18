add_mlir_unittest(MLIRDialectTests
  BroadcastShapeTest.cpp
)
target_link_libraries(MLIRDialectTests
  PRIVATE
  MLIRIR
  MLIRDialect)

add_subdirectory(ArmSME)
add_subdirectory(Index)
add_subdirectory(LLVMIR)
add_subdirectory(MemRef)
add_subdirectory(SCF)
add_subdirectory(SparseTensor)
add_subdirectory(SPIRV)
add_subdirectory(Transform)
add_subdirectory(Utils)

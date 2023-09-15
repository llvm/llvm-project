// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @joint_matrix_load(%ptr : !llvm.ptr<i32>, %stride : index) {
  // expected-error @+1 {{'genx.matrix.load' op scope attribute must have value 'Subgroup'}}
  %0 = genx.matrix.load <Workgroup> <RowMajor> %ptr, %stride {memory_access = #genx.memory_access<Volatile>} : (!llvm.ptr<i32>, index) -> !genx.jointmatrix<8x16xi32, RowMajor>
  llvm.return
}

// -----

func.func @joint_matrix_load(%ptr : !llvm.ptr<i32>, %stride : index) {
  // expected-error @+1 {{'genx.matrix.load' op result layout must match layout attribute}}
  %1 = genx.matrix.load <Subgroup> <RowMajor> %ptr, %stride {memory_access = #genx.memory_access<Volatile>} : (!llvm.ptr<i32>, index) -> !genx.jointmatrix<8x16xi32, ColumnMajor>
  llvm.return
}

// -----

func.func @joint_matrix_init(%mat : !genx.jointmatrix<8x16xi32, RowMajor>, %val : f32) {
  // expected-error @+1 {{'genx.matrix.init' op scope attribute must have value 'Subgroup'}}
  genx.matrix.init <Workgroup> %mat, %val : (!genx.jointmatrix<8x16xi32, RowMajor>, f32)
  llvm.return
}

// -----

func.func @joint_matrix_init(%mat : !genx.jointmatrix<8x16xi32, RowMajor>, %val : f32) {
  // expected-error @+1 {{'genx.matrix.init' op initializer type must match matrix element type}}
  genx.matrix.init <Subgroup> %mat, %val : (!genx.jointmatrix<8x16xi32, RowMajor>, f32)
  llvm.return
}

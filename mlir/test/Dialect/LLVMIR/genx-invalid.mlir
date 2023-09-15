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

func.func @joint_matrix_store(%ptr : !llvm.ptr<i32>, %val: !genx.jointmatrix<8x16xi32, RowMajor>, %stride : index) {
  // expected-error @+1 {{'genx.matrix.store' op scope attribute must have value 'Subgroup'}}  
  genx.matrix.store <Workgroup> <RowMajor> %ptr, %val, %stride {memory_access = #genx.memory_access<Volatile>} : (!llvm.ptr<i32>, !genx.jointmatrix<8x16xi32, RowMajor>, index)
  llvm.return
}

// -----

func.func @joint_matrix_store(%ptr : !llvm.ptr<i32>, %val: !genx.jointmatrix<8x16xi32, ColumnMajor>, %stride : index) {
  // expected-error @+1 {{'genx.matrix.store' op layout of value to store must match layout attribute}}  
  genx.matrix.store <Subgroup> <RowMajor> %ptr, %val, %stride {memory_access = #genx.memory_access<Volatile>} : (!llvm.ptr<i32>, !genx.jointmatrix<8x16xi32, ColumnMajor>, index)
  llvm.return
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xi32, RowMajor>, %b : !genx.jointmatrix<8x8xi32, RowMajor>, %c : !genx.jointmatrix<8x8xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op scope attribute must have value 'Subgroup'}}
  %r = genx.matrix.mad <Workgroup> %a, %b, %c : !genx.jointmatrix<8x16xi32, RowMajor>, !genx.jointmatrix<8x8xi32, RowMajor>, !genx.jointmatrix<8x8xi32, RowMajor> -> !genx.jointmatrix<8x8xi32, RowMajor>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xi32, RowMajor>, %b : !genx.jointmatrix<8x8xi32, RowMajor>, %c : !genx.jointmatrix<8x8xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op failed to verify that all of {c, res} have same type}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xi32, RowMajor>, !genx.jointmatrix<8x8xi32, RowMajor>, !genx.jointmatrix<8x8xi32, RowMajor> -> !genx.jointmatrix<8x16xi32, RowMajor>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xi32, RowMajor>, %b : !genx.jointmatrix<16x8xi32, RowMajor>, %c : !genx.jointmatrix<16x8xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op matrix sizes must match}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xi32, RowMajor>, !genx.jointmatrix<16x8xi32, RowMajor>, !genx.jointmatrix<16x8xi32, RowMajor> -> !genx.jointmatrix<16x8xi32, RowMajor>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<16x16xi32, RowMajor>, %b : !genx.jointmatrix<16x8xi32, RowMajor>, %c : !genx.jointmatrix<16x8xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op result matrix must have a max of 8 rows}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<16x16xi32, RowMajor>, !genx.jointmatrix<16x8xi32, RowMajor>, !genx.jointmatrix<16x8xi32, RowMajor> -> !genx.jointmatrix<16x8xi32, RowMajor>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xi32, RowMajor>, %b : !genx.jointmatrix<16x8xi32, RowMajor>, %c : !genx.jointmatrix<8x8xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op result matrix must have 16 columns}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xi32, RowMajor>, !genx.jointmatrix<16x8xi32, RowMajor>, !genx.jointmatrix<8x8xi32, RowMajor> -> !genx.jointmatrix<8x8xi32, RowMajor>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xi32, RowMajor>, %b : !genx.jointmatrix<16x16xi32, RowMajor>, %c : !genx.jointmatrix<8x16xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op 1st operand matrix must have 32 columns}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xi32, RowMajor>, !genx.jointmatrix<16x16xi32, RowMajor>, !genx.jointmatrix<8x16xi32, RowMajor> -> !genx.jointmatrix<8x16xi32, RowMajor>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x32xf16, RowMajor>, %b : !genx.jointmatrix<32x16xf16, RowMajor>, %c : !genx.jointmatrix<8x16xf16, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op 1st operand matrix must have 16 columns}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x32xf16, RowMajor>, !genx.jointmatrix<32x16xf16, RowMajor>, !genx.jointmatrix<8x16xf16, RowMajor> -> !genx.jointmatrix<8x16xf16, RowMajor>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xf32, RowMajor>, %b : !genx.jointmatrix<16x16xi32, RowMajor>, %c : !genx.jointmatrix<8x16xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op matrix element types must match}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xf32, RowMajor>, !genx.jointmatrix<16x16xi32, RowMajor>, !genx.jointmatrix<8x16xi32, RowMajor> -> !genx.jointmatrix<8x16xi32, RowMajor>
  llvm.return
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x32xi32, RowMajor>, %b : !genx.jointmatrix<32x16xi32, RowMajor>, %c : !genx.jointmatrix<8x16xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op 1st operand element type must have bit-width equal to 8}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x32xi32, RowMajor>, !genx.jointmatrix<32x16xi32, RowMajor>, !genx.jointmatrix<8x16xi32, RowMajor> -> !genx.jointmatrix<8x16xi32, RowMajor>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x32xi8, RowMajor>, %b : !genx.jointmatrix<32x16xi8, RowMajor>, %c : !genx.jointmatrix<8x16xi8, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op 3rd operand element type must be i32}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x32xi8, RowMajor>, !genx.jointmatrix<32x16xi8, RowMajor>, !genx.jointmatrix<8x16xi8, RowMajor> -> !genx.jointmatrix<8x16xi8, RowMajor>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xf32, RowMajor>, %b : !genx.jointmatrix<16x16xf32, RowMajor>, %c : !genx.jointmatrix<8x16xf32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op 1st operand element type must be f16 or bf16}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xf32, RowMajor>, !genx.jointmatrix<16x16xf32, RowMajor>, !genx.jointmatrix<8x16xf32, RowMajor> -> !genx.jointmatrix<8x16xf32, RowMajor>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xf16, RowMajor>, %b : !genx.jointmatrix<16x16xf16, RowMajor>, %c : !genx.jointmatrix<8x16xf16, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op 3rd operand element type must be f32}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xf16, RowMajor>, !genx.jointmatrix<16x16xf16, RowMajor>, !genx.jointmatrix<8x16xf16, RowMajor> -> !genx.jointmatrix<8x16xf16, RowMajor>
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

// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @genx.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'genx.matrix.dpas' op expecting repeat count to be 1, 2, 4, or 8}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=6:i32} : (vector<8xi32>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'genx.matrix.dpas' op expecting precision of matrix A and B to be the same}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<U8>, rc=8:i32} : (vector<8xi32>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi8>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'genx.matrix.dpas' op 1st operand (C) and result (D) should have the same type}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=8:i32} : (vector<8xi8>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<16xi32>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'genx.matrix.dpas' op the dimension for 1st operand (C) and result (D) should match repeat count}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=8:i32} : (vector<16xi32>, vector<16xi8>, vector<32xi8>) -> vector<16xi32>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<8xi32>) {
  // expected-error @+1 {{'genx.matrix.dpas' op element type of 2nd (A) and 3rd (B) operands must match}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=8:i32} : (vector<8xi32>, vector<16xi8>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi32>, %a : vector<8xi8>, %b : vector<8xi8>) {
  // expected-error @+1 {{'genx.matrix.dpas' op 2nd operand (A) bit-size should be repeat count times 16}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=8:i32} : (vector<8xi32>, vector<8xi8>, vector<8xi8>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<16xi8>) {
  // expected-error @+1 {{'genx.matrix.dpas' op 3rd operand (B) bit-size should be systolic depth (8) times 32}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=8:i32} : (vector<8xi32>, vector<16xi8>, vector<16xi8>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi32>, %a : vector<16xsi8>, %b : vector<32xsi8>) {
  // expected-error @+1 {{'genx.matrix.dpas' op precision should be S8 when 2nd (A) or 3rd (B) operand element type is signed i8}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<U8>, pb=#genx.precision_type<U8>, rc=8:i32} : (vector<8xi32>, vector<16xsi8>, vector<32xsi8>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi8>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'genx.matrix.dpas' op the element type for 1st operand (C) and the result should be i32}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=8:i32} : (vector<8xi8>, vector<16xi8>, vector<32xi8>) -> vector<8xi8>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi32>, %a : vector<4xi32>, %b : vector<8xi32>) {
  // expected-error @+1 {{'genx.matrix.dpas' op expecting 2nd (A) or 3rd (B) operand element type to be f32, bf16, f16, or i8}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=8:i32} : (vector<8xi32>, vector<4xi32>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @matrix_2Dblockload(%ptr : !llvm.ptr<i32>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'genx.matrix.2Dblockload' op expecting 'elem_size_in_bits' to be 8, 16, or 32}}
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=64:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr<i32>, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

func.func @matrix_2Dblockload(%ptr : !llvm.ptr<i32>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'genx.matrix.2Dblockload' op transpose and vnni transform are mutually exclusive}}
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=true, vnni_transform=true} : (!llvm.ptr<i32>, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

func.func @matrix_2Dblockload(%ptr : !llvm.ptr<i32>, %base_height : i32, %x : i32, %y : i32) {
  %base_width = llvm.mlir.constant(4 : i32) : i32
  %base_pitch = llvm.mlir.constant(2 : i32) : i32
  // expected-error @+1 {{'genx.matrix.2Dblockload' op 4th operand (base pitch) should be >= 2nd operand (base width)}}
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr<i32>, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

func.func @matrix_2Dblockstore(%ptr : !llvm.ptr<i32>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  // expected-error @+1 {{'genx.matrix.2Dblockstore' op expecting 'elem_size_in_bits' to be 8, 16, or 32}}
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=64:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr<i32>, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}

// -----

func.func @matrix_2Dblockstore(%ptr : !llvm.ptr<i32>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  // expected-error @+1 {{'genx.matrix.2Dblockstore' op transpose and vnni transform are mutually exclusive}}
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=true, vnni_transform=true} : (!llvm.ptr<i32>, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}

// -----

func.func @matrix_2Dblockstore(%ptr : !llvm.ptr<i32>, %base_height : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  %base_width = llvm.mlir.constant(4 : i32) : i32
  %base_pitch = llvm.mlir.constant(2 : i32) : i32
  // expected-error @+1 {{'genx.matrix.2Dblockstore' op 4th operand (base pitch) should be >= 2nd operand (base width)}}
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr<i32>, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}

// -----

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

// -----

func.func @joint_matrix_copy(%src : !genx.jointmatrix<8x16xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.copy' op scope attribute must have value 'Subgroup'}}
  %0 = genx.matrix.copy <Workgroup> %src : (!genx.jointmatrix<8x16xi32, RowMajor>) -> !genx.jointmatrix<8x16xi32, RowMajor>
  llvm.return
}

// -----

func.func @joint_matrix_copy(%src : !genx.jointmatrix<8x16xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.copy' op result shape must match source shape}}
  %0 = genx.matrix.copy <Subgroup> %src : (!genx.jointmatrix<8x16xi32, RowMajor>) -> !genx.jointmatrix<16x8xi32, RowMajor>
  llvm.return
}

// -----

func.func @joint_matrix_copy(%src : !genx.jointmatrix<8x16xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.copy' op result layout must match source layout}}
  %0 = genx.matrix.copy <Subgroup> %src : (!genx.jointmatrix<8x16xi32, RowMajor>) -> !genx.jointmatrix<8x16xf32, ColumnMajor>
  llvm.return
}

// -----

func.func @joint_matrix_mad(%mat : !genx.jointmatrix<8x16xf32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.map' op scope attribute must have value 'Subgroup'}}
  %r = genx.matrix.map <Workgroup> 
    ins(%mat : !genx.jointmatrix<8x16xf32, RowMajor>)
    (%elem: f32) {
       %cst = arith.constant 1.0 : f32
       %add = arith.addf %elem, %cst : f32
       genx.yield %add : f32
    } : !genx.jointmatrix<8x16xf32, RowMajor>  
  llvm.return
}

// -----

func.func @joint_matrix_mad(%mat : !genx.jointmatrix<8x16xf32, RowMajor>, %val: f32) {
  // expected-error @+1 {{'genx.matrix.map' op expected element type of input 'f32' to match bbArg type 'i32'}}
  %r = genx.matrix.map <Subgroup> 
    ins(%mat, %val : !genx.jointmatrix<8x16xf32, RowMajor>, f32)
    (%elem: i32, %v: i32) {
       %add = arith.addf %elem, %v : i32
       genx.yield %add : i32
    } : !genx.jointmatrix<8x16xf32, RowMajor>  
  llvm.return
}

// -----

func.func @joint_matrix_mad(%mat : !genx.jointmatrix<8x16xf32, RowMajor>, %val: f32) {
  // expected-error @+1 {{'genx.matrix.map' op expects number of operands to match the arity of mapper, but got: 2 and 1}}
  %r = genx.matrix.map <Subgroup> 
    ins(%mat, %val : !genx.jointmatrix<8x16xf32, RowMajor>, f32)
    (%elem: f32) {
       %cst = arith.constant 1.0 : f32
       %add = arith.addf %elem, %cst : f32
       genx.yield %add : f32
    } : !genx.jointmatrix<8x16xf32, RowMajor>  
  llvm.return
}

// -----

func.func @joint_matrix_mad(%mat : !genx.jointmatrix<8x16xf32, RowMajor>, %val: f32) {
  // expected-error @+1 {{'genx.matrix.map' op expected type of input 'f32' to match bbArg type 'i32'}}
  %r = genx.matrix.map <Subgroup> 
    ins(%mat, %val : !genx.jointmatrix<8x16xf32, RowMajor>, f32)
    (%elem: f32, %v: i32) {
       %cast = llvm.bitcast %v : i32 to f32
       %add = arith.addf %elem, %cast : f32
       genx.yield %add : f32
    } : !genx.jointmatrix<8x16xf32, RowMajor>  
  llvm.return
}

// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @tensormap_replace_missing_ordinal_1(%addr : !llvm.ptr<1>, %new_val : i32) {
  // expected-error @+1 {{ordinal is required for global_stride field}}
  nvvm.tensormap.replace field = global_stride, new_value = %new_val in %addr : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @tensormap_replace_missing_ordinal_2(%addr : !llvm.ptr<1>, %new_val : i32) {
  // expected-error @+1 {{ordinal is required for box_dim field}}
  nvvm.tensormap.replace field = box_dim, new_value = %new_val in %addr : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @tensormap_replace_invalid_ordinal(%addr : !llvm.ptr<1>, %new_val : i32) {
  // expected-error @+1 {{ordinal is not supported for rank field}}
  nvvm.tensormap.replace field = rank[1], new_value = %new_val in %addr : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @tensormap_replace_ordinal_out_of_range(%addr : !llvm.ptr<1>, %new_val : i32) {
  // expected-error @+1 {{attribute 'ord' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 1 whose maximum value is 5}}
  nvvm.tensormap.replace field = box_dim[6], new_value = %new_val in %addr : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @tensormap_replace_missing_new_val(%addr : !llvm.ptr<1>) {
  // expected-error @+1 {{new_value must be specified and must be an i32 for rank field}}
  nvvm.tensormap.replace field = rank, new_value = in %addr : !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tensormap_replace_invalid_new_val_1(%addr : !llvm.ptr<1>, %new_val : i64) {
  // expected-error @+1 {{new_value must be specified and must be an i32 for rank field}}
  nvvm.tensormap.replace field = rank, new_value = %new_val in %addr : !llvm.ptr<1>, i64
  llvm.return
}

// -----

llvm.func @tensormap_replace_invalid_new_val_2(%addr : !llvm.ptr<1>, %new_val : i64) {
  // expected-error @+1 {{new_value must be specified and must be an i32 for box_dim field}}
  nvvm.tensormap.replace field = box_dim[1], new_value = %new_val in %addr : !llvm.ptr<1>, i64
  llvm.return
}

// -----

llvm.func @tensormap_replace_missing_new_val_attr(%addr : !llvm.ptr<1>) {
  // expected-error @+1 {{new_value_attr must be specified and must be a valid elemtype attribute for elemtype field}}
  nvvm.tensormap.replace field = elemtype, new_value = in %addr : !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tensormap_replace_invalid_new_val_attr(%addr : !llvm.ptr<1>, %new_val : i32) {
  // expected-error @+1 {{new_value_attr must be specified and must be a valid elemtype attribute for elemtype field}}
  nvvm.tensormap.replace field = elemtype, new_value = %new_val in %addr : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @tensormap_replace_invalid_global_address(%addr : !llvm.ptr<1>, %new_val : i32) {
  // expected-error @+1 {{new_value must be specified and must be an i64 for global_address field}}
  nvvm.tensormap.replace field = global_address, new_value = %new_val in %addr : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @tensormap_replace_invalid_global_stride(%addr : !llvm.ptr<1>, %new_val : i32) {
  // expected-error @+1 {{new_value must be specified and must be an i64 for global_stride field}}
  nvvm.tensormap.replace field = global_stride[1], new_value = %new_val in %addr : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @tensormap_replace_invalid_elemtype(%addr : !llvm.ptr<1>) {
  // expected-error @+1 {{new_value_attr must be specified and must be a valid elemtype attribute for elemtype field}}
  nvvm.tensormap.replace field = elemtype, new_value = #nvvm.tensormap_interleave_layout<b16> in %addr : !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tensormap_replace_invalid_interleave_layout(%addr : !llvm.ptr<1>) {
  // expected-error @+1 {{new_value_attr must be specified and must be a valid interleave_layout attribute for interleave_layout field}}
  nvvm.tensormap.replace field = interleave_layout, new_value = #nvvm.tensormap_swizzle_mode<b32> in %addr : !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tensormap_replace_invalid_swizzle_mode(%addr : !llvm.ptr<1>) {
  // expected-error @+1 {{new_value_attr must be specified and must be a valid swizzle_mode attribute for swizzle_mode field}}
  nvvm.tensormap.replace field = swizzle_mode, new_value = #nvvm.tensormap_swizzle_atomicity<b32> in %addr : !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tensormap_replace_invalid_swizzle_atomicity(%addr : !llvm.ptr<1>) {
  // expected-error @+1 {{new_value_attr must be specified and must be a valid swizzle_atomicity attribute for swizzle_atomicity field}}
  nvvm.tensormap.replace field = swizzle_atomicity, new_value = #nvvm.tensormap_fill_mode<zero> in %addr : !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tensormap_replace_invalid_fill_mode(%addr : !llvm.ptr<1>) {
  // expected-error @+1 {{new_value_attr must be specified and must be a valid fill_mode attribute for fill_mode field}}
  nvvm.tensormap.replace field = fill_mode, new_value = #nvvm.tensormap_elemtype<s32> in %addr : !llvm.ptr<1>
  llvm.return
}

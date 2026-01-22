// RUN: mlir-translate --mlir-to-llvmir -verify-diagnostics -split-input-file %s

llvm.func @tcgen05_ld_red_same_types(%addr : !llvm.ptr<6>) {
  // expected-error @below {{type of reduction value and element type of vector data should match}}
  %data, %redval = nvvm.tcgen05.ld.red %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, op = #nvvm.tcgen05_ld_red_op<min>} : vector<2 x i32>, f32
  llvm.return
}

// -----

llvm.func @tcgen05_ld_red_i32_abs_nan(%addr : !llvm.ptr<6>) {
  // expected-error @below {{abs or nan is only applicable for f32 type}}
  %data, %redval = nvvm.tcgen05.ld.red %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, op = #nvvm.tcgen05_ld_red_op<min>, nan, abs} : vector<2 x i32>, i32
  llvm.return
}

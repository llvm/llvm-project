// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @nvvm_tcgen05_ld_32x32b_offset(%tmemAddr : !llvm.ptr<6>, %offset : i64) -> () {
  // expected-error@+1 {{offset argument is only supported for shape 16x32bx2}}
  %ldv2 = nvvm.tcgen05.ld %tmemAddr, %offset { pack, shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<2 x i32>
  llvm.return
}

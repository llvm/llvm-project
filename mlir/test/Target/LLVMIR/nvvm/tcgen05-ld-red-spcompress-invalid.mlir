// RUN: mlir-translate --mlir-to-llvmir -verify-diagnostics -split-input-file %s

llvm.func @tcgen05_ld_red_spcompress_add(%addr : !llvm.ptr<6>) {
  // expected-error @below {{'nvvm.tcgen05.ld.red.spcompress' op attribute 'op' failed to satisfy constraint: NVVM Reduction Kind attribute whose value is one of {min, max}}}
  %m, %d, %r = nvvm.tcgen05.ld.red.spcompress add %addr : i32, vector<2xf32>, f32
  llvm.return
}

// -----

llvm.func @tcgen05_ld_red_spcompress_i32_data(%addr : !llvm.ptr<6>) {
  // expected-error @below {{'nvvm.tcgen05.ld.red.spcompress' op result #1 must be fixed-length vector of 32-bit float values of length 2/4/8/16/32/64 of ranks 1}}
  %m, %d, %r = nvvm.tcgen05.ld.red.spcompress min %addr : i32, vector<2xi32>, f32
  llvm.return
}

// -----

llvm.func @tcgen05_ld_red_spcompress_metadata_mismatch(%addr : !llvm.ptr<6>) {
  // expected-error @below {{'nvvm.tcgen05.ld.red.spcompress' op expects metadata to occupy 2 32-bit register(s) for repeat factor x64}}
  %m, %d, %r = nvvm.tcgen05.ld.red.spcompress min %addr nan = true : i32, vector<32xf32>, f32
  llvm.return
}

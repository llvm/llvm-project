// RUN: mlir-translate --mlir-to-llvmir -verify-diagnostics -split-input-file %s

llvm.func @nvvm_tcgen05_ld_spcompress_add(%addr : !llvm.ptr<6>) {
  // expected-error@+1 {{'nvvm.tcgen05.ld.red.spcompress' op attribute 'op' failed to satisfy constraint: NVVM Reduction Kind attribute whose value is one of {min, max}}}
  %m, %d = nvvm.tcgen05.ld.red.spcompress add %addr : i32, vector<2xf32>
  llvm.return
}

// -----

llvm.func @nvvm_tcgen05_ld_spcompress_x2(%addr : !llvm.ptr<6>) {
  // expected-error@+1 {{'nvvm.tcgen05.ld.red.spcompress' op result #1 must be fixed-length vector of 32-bit float values of length 2/4/8/16/32/64 of ranks 1}}
  %m, %d = nvvm.tcgen05.ld.red.spcompress min %addr : i32, vector<1xf32>
  llvm.return
}

// -----

llvm.func @nvvm_tcgen05_ld_spcompress_metadata_too_large(%addr : !llvm.ptr<6>) {
  // expected-error@+1 {{'nvvm.tcgen05.ld.red.spcompress' op expects metadata to occupy 1 32-bit register(s) for repeat factor x32}}
  %m, %d = nvvm.tcgen05.ld.red.spcompress min %addr : vector<2xi32>, vector<16xf32>
  llvm.return
}

// -----

llvm.func @nvvm_tcgen05_ld_spcompress_metadata_too_small(%addr : !llvm.ptr<6>) {
  // expected-error@+1 {{'nvvm.tcgen05.ld.red.spcompress' op expects metadata to occupy 4 32-bit register(s) for repeat factor x128}}
  %m, %d = nvvm.tcgen05.ld.red.spcompress max %addr abs = true : vector<2xi32>, vector<64xf32>
  llvm.return
}

// -----

llvm.func @nvvm_tcgen05_ld_spcompress_nan_without_redval(%addr : !llvm.ptr<6>) {
  // expected-error@+1 {{'nvvm.tcgen05.ld.red.spcompress' op nan requires the reduction result redVal}}
  %m, %d = nvvm.tcgen05.ld.red.spcompress min %addr nan = true : i32, vector<2xf32>
  llvm.return
}

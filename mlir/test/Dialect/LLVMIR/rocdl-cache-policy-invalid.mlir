// RUN: mlir-opt %s -split-input-file -verify-diagnostics

llvm.func @pre_gfx12_op_rejects_gfx12(%src : !llvm.ptr<1>,
                                      %dst : !llvm.ptr<3>) {
  // expected-error@+1 {{failed to satisfy constraint: pre-gfx12 or gfx942 AMDGPU cache policy attribute}}
  rocdl.global.load.async.lds %src, %dst, 4, 0, gfx12<nt> : !llvm.ptr<1>, !llvm.ptr<3>
  llvm.return
}

// -----

llvm.func @gfx12_op_rejects_pre_gfx12(%src : !llvm.ptr<1>,
                                      %dst : !llvm.ptr<3>) {
  // expected-error@+1 {{failed to satisfy constraint: gfx12 non-atomic AMDGPU cache policy attribute}}
  rocdl.global.load.async.to.lds.b32 %src, %dst, 0, pre_gfx12<glc> : !llvm.ptr<1>, !llvm.ptr<3>
  llvm.return
}

// -----

llvm.func @non_atomic_buffer_rejects_gfx12_atomic(%rsrc : vector<4xi32>,
                                                 %offset : i32,
                                                 %soffset : i32) {
  // expected-error@+1 {{failed to satisfy constraint: non-atomic AMDGPU buffer cache policy attribute}}
  %0 = rocdl.raw.buffer.load %rsrc, %offset, %soffset, gfx12_atomic<nt> : i32
  llvm.return
}

// -----

llvm.func @atomic_buffer_rejects_gfx12(%rsrc : vector<4xi32>,
                                       %offset : i32,
                                       %soffset : i32,
                                       %vdata : i32) {
  // expected-error@+1 {{failed to satisfy constraint: atomic AMDGPU buffer cache policy attribute}}
  %0 = rocdl.raw.buffer.atomic.smax %vdata, %rsrc, %offset, %soffset, gfx12<nt> : i32
  llvm.return
}

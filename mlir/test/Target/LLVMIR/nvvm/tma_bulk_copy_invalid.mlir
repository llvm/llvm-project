// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

llvm.func @tma_bulk_copy_g2s_mc(%src : !llvm.ptr<1>, %dest : !llvm.ptr<3>, %bar : !llvm.ptr<3>, %size : i32, %ctamask : i16) {
  // expected-error @below {{Multicast is not supported with shared::cta mode.}}
  nvvm.cp.async.bulk.shared.cluster.global %dest, %src, %bar, %size multicast_mask = %ctamask : !llvm.ptr<3>, !llvm.ptr<1>

  llvm.return
}

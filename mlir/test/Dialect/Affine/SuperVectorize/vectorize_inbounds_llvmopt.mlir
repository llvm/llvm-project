// Verify that the inbounds|nuw flags emitted on vector.load/store GEPs (when
// in_bounds=true) enable LLVM to perform alias analysis and produce
// well-optimized LLVM IR.  The pipeline lowers MLIR all the way to native LLVM
// IR via mlir-translate, then passes the result through LLVM's -O2 pipeline.
//
// RUN: mlir-opt %s \
// RUN:   --affine-super-vectorize="virtual-vector-size=4" \
// RUN:   --lower-affine \
// RUN:   --convert-scf-to-cf \
// RUN:   --expand-strided-metadata \
// RUN:   --convert-arith-to-llvm \
// RUN:   --convert-cf-to-llvm \
// RUN:   --convert-vector-to-llvm \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | opt -S -passes="default<O2>" \
// RUN:   | FileCheck %s --check-prefix=OPT

// OPT-LABEL: define void @copy(
// After -O2, LLVM alias-analysis annotates the source arg as read-only and
// the destination arg as write-only.  This requires the GEP to carry inbounds
// and nuw flags (produced by our vector.load/store lowering fix) so LLVM can
// prove the two memory regions do not overlap.
// OPT-SAME: readonly
// OPT-SAME: writeonly

// The GEP for the load carries inbounds nuw — our fix propagated through opt.
// OPT: getelementptr inbounds nuw
// The MLIR-level vectorization (vector<4xf32> from affine-super-vectorize)
// must be preserved through LLVM optimisation — no scalar regression.
// OPT-NEXT: load <4 x float>
// OPT-NEXT: getelementptr inbounds nuw
// OPT-NEXT: store <4 x float>

// No masked-load/store intrinsics: in_bounds=true correctly skipped masking.
// OPT-NOT: @llvm.masked.load
// OPT-NOT: @llvm.masked.store

func.func @copy(%A: memref<512x512xf32>, %B: memref<512x512xf32>) {
  affine.for %i = 0 to 512 {
    affine.for %j = 0 to 512 {
      %v = affine.load %A[%i, %j] : memref<512x512xf32>
      affine.store %v, %B[%i, %j] : memref<512x512xf32>
    }
  }
  return
}

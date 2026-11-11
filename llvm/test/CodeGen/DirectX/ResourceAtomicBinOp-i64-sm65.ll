; RUN: not opt -S -dxil-resource-access -dxil-op-lower -mtriple=dxil-pc-shadermodel6.5-compute %s 2>&1 | FileCheck %s

; Verify resource i64 atomicrmw rejects shader models before SM 6.6, where
; dx.op.atomicBinOp gained i64 overload support.

target triple = "dxil-pc-shadermodel6.5-compute"

define i64 @atomic_i64(i32 %index, i64 %value) {
  %buffer = call target("dx.RawBuffer", i64, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i64, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: Cannot create AtomicBinOp operation: Invalid overload type
  %old = atomicrmw add ptr %ptr, i64 %value monotonic
  ret i64 %old
}

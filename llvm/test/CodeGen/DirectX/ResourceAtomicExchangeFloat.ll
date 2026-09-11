; RUN: opt -S -dxil-resource-access -dxil-op-lower -mtriple=dxil-pc-shadermodel6.0-compute %s | FileCheck %s

; DXIL has no floating-point atomic op. A float exchange only moves the bit
; pattern, so it lowers to an i32 AtomicBinOp with a bitcast on the value and
; on the returned original value. This needs no capability bits, so it works
; from SM 6.0.

target triple = "dxil-pc-shadermodel6.0-compute"

; CHECK-LABEL: define float @bab_xchg_float
define float @bab_xchg_float(i32 %offset, float %val) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer, i32 %offset)
  ; CHECK: [[CAST:%.*]] = bitcast float %val to i32
  ; CHECK: [[OLD:%.*]] = call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 8, i32 %offset, i32 poison, i32 0, i32 [[CAST]])
  ; CHECK: [[RES:%.*]] = bitcast i32 [[OLD]] to float
  %old = atomicrmw xchg ptr %ptr, float %val monotonic
  ; CHECK: ret float [[RES]]
  ret float %old
}

; A StructuredBuffer of float keeps the struct index in coord0 and the byte
; offset in coord1.
; CHECK-LABEL: define float @sbuf_xchg_float
define float @sbuf_xchg_float(i32 %index, float %val) {
  %buffer = call target("dx.RawBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", float, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: [[CAST:%.*]] = bitcast float %val to i32
  ; CHECK: [[OLD:%.*]] = call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 8, i32 %index, i32 0, i32 0, i32 [[CAST]])
  ; CHECK: [[RES:%.*]] = bitcast i32 [[OLD]] to float
  %old = atomicrmw xchg ptr %ptr, float %val monotonic
  ; CHECK: ret float [[RES]]
  ret float %old
}

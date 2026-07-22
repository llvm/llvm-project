; RUN: opt -S -dxil-resource-type -dxil-resource-access -disable-verify \
; RUN:  -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Test hoisting a sunk resource store through a phi where the store addresses
; the resource through a GEP chain and the stored value is itself a phi. The
; GEP chain and the value are rematerialized into each predecessor: following
; the pointer phi's per-edge value naturally follows the value phi's matching
; per-edge value.

@In0.str = internal unnamed_addr constant [4 x i8] c"In0\00", align 1
@In1.str = internal unnamed_addr constant [4 x i8] c"In1\00", align 1

; CHECK-LABEL: @main(
define void @main(i32 %idx, double %v0, double %v1) {
entry:
  %in0 = tail call target("dx.RawBuffer", <4 x double>, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f64_0_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @In0.str)
  %in1 = tail call target("dx.RawBuffer", <4 x double>, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f64_0_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @In1.str)
  %tid = tail call i32 @llvm.dx.thread.id(i32 0)
  %cmp = icmp eq i32 %tid, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
; CHECK:      if.then:
; CHECK-NEXT:   %[[OFF0:.*]] = mul i32 %idx, 8
; CHECK-NEXT:   call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v4f64_0_0t.f64(target("dx.RawBuffer", <4 x double>, 0, 0) %in0, i32 %tid, i32 %[[OFF0]], double %v0)
; CHECK-NEXT:   br label %exit
  %p0 = call noundef nonnull ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_v4f64_0_0t(target("dx.RawBuffer", <4 x double>, 0, 0) %in0, i32 %tid)
  br label %exit

if.else:
; CHECK:      if.else:
; CHECK-NEXT:   %[[OFF1:.*]] = mul i32 %idx, 8
; CHECK-NEXT:   call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v4f64_0_0t.f64(target("dx.RawBuffer", <4 x double>, 0, 0) %in1, i32 %tid, i32 %[[OFF1]], double %v1)
; CHECK-NEXT:   br label %exit
  %p1 = call noundef nonnull ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_v4f64_0_0t(target("dx.RawBuffer", <4 x double>, 0, 0) %in1, i32 %tid)
  br label %exit

exit:
; CHECK:      exit:
; CHECK-NOT:    phi ptr
; CHECK-NOT:    phi double
; CHECK-NEXT:   ret void
  %ptr = phi ptr [ %p0, %if.then ], [ %p1, %if.else ]
  %vphi = phi double [ %v0, %if.then ], [ %v1, %if.else ]
  %gep = getelementptr double, ptr %ptr, i32 %idx
  store double %vphi, ptr %gep, align 8
  ret void
}

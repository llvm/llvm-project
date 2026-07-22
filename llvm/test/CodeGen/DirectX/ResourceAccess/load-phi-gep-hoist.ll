; RUN: opt -S -dxil-resource-type -dxil-resource-access -disable-verify \
; RUN:  -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Test that we can resolve the case when InstCombine has sunk a load of a
; resource through a phi node where the load addresses the resource through a
; GEP chain. The GEP chain rooted at the phi must be duplicated into each
; predecessor along with the hoisted load. This is required due to the
; inability of preventing the sink using available memory semantics.

@In0.str = internal unnamed_addr constant [4 x i8] c"In0\00", align 1
@In1.str = internal unnamed_addr constant [4 x i8] c"In1\00", align 1

; CHECK-LABEL: @main(
define void @main(i32 %idx) {
entry:
  %in0 = tail call target("dx.RawBuffer", <4 x double>, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f64_0_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @In0.str)
  %in1 = tail call target("dx.RawBuffer", <4 x double>, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f64_0_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @In1.str)
  %tid = tail call i32 @llvm.dx.thread.id(i32 0)
  %cmp = icmp eq i32 %tid, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
; CHECK:      if.then:
; CHECK-NEXT:   %[[OFF0:.*]] = mul i32 %idx, 8
; CHECK-NEXT:   %[[LOAD0:.*]] = call { double, i1 } @llvm.dx.resource.load.rawbuffer.f64.tdx.RawBuffer_v4f64_0_0t(target("dx.RawBuffer", <4 x double>, 0, 0) %in0, i32 %tid, i32 %[[OFF0]])
; CHECK-NEXT:   %[[VAL0:.*]] = extractvalue { double, i1 } %[[LOAD0]], 0
; CHECK-NEXT:   br label %exit
  %p0 = call noundef nonnull ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_v4f64_0_0t(target("dx.RawBuffer", <4 x double>, 0, 0) %in0, i32 %tid)
  br label %exit

if.else:
; CHECK:      if.else:
; CHECK-NEXT:   %[[OFF1:.*]] = mul i32 %idx, 8
; CHECK-NEXT:   %[[LOAD1:.*]] = call { double, i1 } @llvm.dx.resource.load.rawbuffer.f64.tdx.RawBuffer_v4f64_0_0t(target("dx.RawBuffer", <4 x double>, 0, 0) %in1, i32 %tid, i32 %[[OFF1]])
; CHECK-NEXT:   %[[VAL1:.*]] = extractvalue { double, i1 } %[[LOAD1]], 0
; CHECK-NEXT:   br label %exit
  %p1 = call noundef nonnull ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_v4f64_0_0t(target("dx.RawBuffer", <4 x double>, 0, 0) %in1, i32 %tid)
  br label %exit

exit:
; CHECK:      exit:
; CHECK-NEXT:   %[[VAL:.*]] = phi double [ %[[VAL0]], %if.then ], [ %[[VAL1]], %if.else ]
; CHECK-NOT:    phi ptr
; CHECK-NEXT:   call void @double_user(double %[[VAL]])
  %ptr = phi ptr [ %p0, %if.then ], [ %p1, %if.else ]
  %gep = getelementptr double, ptr %ptr, i32 %idx
  %val = load double, ptr %gep, align 8
  call void @double_user(double %val)
  ret void
}

declare void @double_user(double)

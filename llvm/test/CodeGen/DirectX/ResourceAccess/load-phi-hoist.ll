; RUN: opt -S -dxil-resource-type -dxil-resource-access -disable-verify \
; RUN:  -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Test that we can resolve the simple case when InstCombine has sunk a load of
; a resource through a phi node. This results in an illegal looking resource
; access, but can be resolved. This is required due to the inability of
; preventing the sink using available memory semantics.

@In0.str = internal unnamed_addr constant [4 x i8] c"In0\00", align 1
@In1.str = internal unnamed_addr constant [4 x i8] c"In1\00", align 1

; CHECK-LABEL: @main()
define void @main() {
entry:
  %in0 = tail call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @In0.str)
  %tid = tail call i32 @llvm.dx.thread.id(i32 0)
  %cmp = icmp eq i32 %tid, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
; CHECK:      if.then:
; CHECK-NEXT:   %[[LOAD0:.*]] = call { float, i1 } @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %in0, i32 %tid, i32 0)
; CHECK-NEXT:   %[[VAL0:.*]] = extractvalue { float, i1 } %[[LOAD0]], 0
; CHECK-NEXT:   br label %exit
  %p0 = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %in0, i32 %tid)
  br label %exit

if.else:
; CHECK:      if.else:
; CHECK:        %[[LOAD1:.*]] = call { float, i1 } @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %in1, i32 %tid, i32 0)
; CHECK-NEXT:   %[[VAL1:.*]] = extractvalue { float, i1 } %[[LOAD1]], 0
; CHECK-NEXT:   br label %exit
  %in1 = tail call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @In1.str)
  %p1 = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %in1, i32 %tid)
  br label %exit

exit:
; CHECK:      exit:
; CHECK-NEXT:   %[[VAL:.*]] = phi float [ %[[VAL0]], %if.then ], [ %[[VAL1]], %if.else ]
; CHECK-NOT:    phi ptr
; CHECK:        %[[ADD:.*]] = fadd float %{{.*}}, %[[VAL]]
  %ptr = phi ptr [ %p0, %if.then ], [ %p1, %if.else ]
  %val = load float, ptr %ptr, align 4
  %out = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %in0, i32 %tid)
  %cur = load float, ptr %out, align 4
  %add = fadd float %cur, %val
  store float %add, ptr %out, align 4
  ret void
}

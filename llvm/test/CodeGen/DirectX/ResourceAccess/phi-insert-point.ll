; RUN: opt -S -dxil-resource-type -dxil-resource-access \
; RUN:  -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Regression test for handle resolution when the resource pointer being replaced
; is itself a PHI node that shares its block with a sibling PHI surviving the
; transform.

@OutArr.str = internal unnamed_addr constant [7 x i8] c"OutArr\00", align 1

; CHECK-LABEL: ptr_phi_before_sibling_phi(
; CHECK-SAME:   i1 %[[COND:.*]], i32 %[[A:.*]], i32 %[[B:.*]])
define i32 @ptr_phi_before_sibling_phi(i1 %cond, i32 %a, i32 %b) {
; CHECK-NOT: handlefromimplicitbinding
entry:
  %handle0 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 0, ptr nonnull @OutArr.str)
  %ptr0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle0, i32 %a)
  br i1 %cond, label %if.then.i, label %main

if.then.i:
  %handle1 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 1, ptr nonnull @OutArr.str)
  %ptr1 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %handle1, i32 %b)
  br label %main

main:
; CHECK:     main:
; CHECK-NEXT:  %[[C:.*]] = phi i32 [ %[[A]], %entry ], [ %[[B]], %if.then.i ]
; CHECK-NEXT:  %[[IDX:.*]] = phi i32 [ 0, %entry ], [ 1, %if.then.i ]
; CHECK-NEXT:  %[[SIBLING:.*]] = phi i32 [ %[[A]], %entry ], [ %[[B]], %if.then.i ]
; CHECK-NEXT:  %[[HANDLE:.*]] = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 -1, i32 %[[IDX]], ptr nonnull @OutArr.str)
; CHECK-NEXT:  call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %[[HANDLE]], i32 %[[C]], i32 0, i32 %[[SIBLING]])
; CHECK-NEXT:  ret i32 %[[SIBLING]]
  %ptr_phi = phi ptr [ %ptr0, %entry ], [ %ptr1, %if.then.i ]
  %sibling = phi i32 [ %a, %entry ], [ %b, %if.then.i ]
  store i32 %sibling, ptr %ptr_phi, align 4
  ret i32 %sibling
}

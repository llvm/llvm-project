; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; This code causes an assertion failure if dereferenceable flag is not properly set in the load generated for memcpy

; CHECK-LABEL: @func
; CHECK: lxvd2x [[VREG:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOT: lxvd2x
; CHECK: stxvd2x [[VREG:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: stxvd2x [[VREG:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr

define void @func(i1 %flag) {
entry:
  %pairs = alloca [4 x <2 x i64>], align 8
  %pair1 = getelementptr inbounds [4 x <2 x i64>], ptr %pairs, i64 0, i64 1
  %pair2 = getelementptr inbounds [4 x <2 x i64>], ptr %pairs, i64 0, i64 2
  br i1 %flag, label %end, label %dummy

end:
  ; copy third element into first element by memcpy
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 nonnull %pairs, ptr align 8 %pair2, i64 16, i1 false)
  ; copy third element into second element by LD/ST
  %vec2 = load <2 x i64>, ptr %pair2, align 8
  store <2 x i64> %vec2, ptr %pair1, align 8
  ret void

dummy:
  ; to make use of %pair2 in another BB
  call void @llvm.memcpy.p0.p0.i64(ptr %pair2, ptr %pair2, i64 0, i1 false)
  br label %end
}


; CHECK-LABEL: @func2
; CHECK: lxvd2x [[VREG:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOT: lxvd2x
; CHECK: stxvd2x [[VREG:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: stxvd2x [[VREG:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr

define void @func2(i1 %flag) {
entry:
  %pairs = alloca [4 x <2 x i64>], align 8
  %pair1 = getelementptr inbounds [4 x <2 x i64>], ptr %pairs, i64 0, i64 1
  %pair2 = getelementptr inbounds [4 x <2 x i64>], ptr %pairs, i64 0, i64 2
  br i1 %flag, label %end, label %dummy

end:
  ; copy third element into first element by memcpy
  call void @llvm.memmove.p0.p0.i64(ptr align 8 nonnull %pairs, ptr align 8 %pair2, i64 16, i1 false)
  ; copy third element into second element by LD/ST
  %vec2 = load <2 x i64>, ptr %pair2, align 8
  store <2 x i64> %vec2, ptr %pair1, align 8
  ret void

dummy:
  ; to make use of %pair2 in another BB
  call void @llvm.memcpy.p0.p0.i64(ptr %pair2, ptr %pair2, i64 0, i1 false)
  br label %end
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1
declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1

attributes #1 = { argmemonly nounwind }

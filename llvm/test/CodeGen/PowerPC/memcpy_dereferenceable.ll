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
  %pair1 = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* %pairs, i64 0, i64 1
  %pair2 = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* %pairs, i64 0, i64 2
  %pvec1 = bitcast <2 x i64>* %pair1 to <2 x i64>*
  %pvec2 = bitcast <2 x i64>* %pair2 to <2 x i64>*
  %dst = bitcast [4 x <2 x i64>]* %pairs to i8*
  %src = bitcast <2 x i64>* %pair2 to i8*
  br i1 %flag, label %end, label %dummy

end:
  ; copy third element into first element by memcpy
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 nonnull %dst, i8* align 8 %src, i64 16, i1 false)
  ; copy third element into second element by LD/ST
  %vec2 = load <2 x i64>, <2 x i64>* %pvec2, align 8
  store <2 x i64> %vec2, <2 x i64>* %pvec1, align 8
  ret void

dummy:
  ; to make use of %src in another BB
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %src, i8* %src, i64 0, i1 false)
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
  %pair1 = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* %pairs, i64 0, i64 1
  %pair2 = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* %pairs, i64 0, i64 2
  %pvec1 = bitcast <2 x i64>* %pair1 to <2 x i64>*
  %pvec2 = bitcast <2 x i64>* %pair2 to <2 x i64>*
  %dst = bitcast [4 x <2 x i64>]* %pairs to i8*
  %src = bitcast <2 x i64>* %pair2 to i8*
  br i1 %flag, label %end, label %dummy

end:
  ; copy third element into first element by memcpy
  call void @llvm.memmove.p0i8.p0i8.i64(i8* align 8 nonnull %dst, i8* align 8 %src, i64 16, i1 false)
  ; copy third element into second element by LD/ST
  %vec2 = load <2 x i64>, <2 x i64>* %pvec2, align 8
  store <2 x i64> %vec2, <2 x i64>* %pvec1, align 8
  ret void

dummy:
  ; to make use of %src in another BB
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %src, i8* %src, i64 0, i1 false)
  br label %end
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

attributes #1 = { argmemonly nounwind }

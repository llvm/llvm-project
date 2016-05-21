; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-p1:32:32:32-p2:16:16:16-n8:16:32:64"


declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p1i8.i32(i8*, i8 addrspace(1)*, i32, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p2i8.i32(i8*, i8 addrspace(2)*, i32, i32, i1) nounwind


define i32* @combine_redundant_addrspacecast(i32 addrspace(1)* %x) nounwind {
; CHECK-LABEL: @combine_redundant_addrspacecast(
; CHECK: addrspacecast i32 addrspace(1)* %x to i32*
; CHECK-NEXT: ret
  %y = addrspacecast i32 addrspace(1)* %x to i32 addrspace(3)*
  %z = addrspacecast i32 addrspace(3)* %y to i32*
  ret i32* %z
}

define <4 x i32*> @combine_redundant_addrspacecast_vector(<4 x i32 addrspace(1)*> %x) nounwind {
; CHECK-LABEL: @combine_redundant_addrspacecast_vector(
; CHECK: addrspacecast <4 x i32 addrspace(1)*> %x to <4 x i32*>
; CHECK-NEXT: ret
  %y = addrspacecast <4 x i32 addrspace(1)*> %x to <4 x i32 addrspace(3)*>
  %z = addrspacecast <4 x i32 addrspace(3)*> %y to <4 x i32*>
  ret <4 x i32*> %z
}

define float* @combine_redundant_addrspacecast_types(i32 addrspace(1)* %x) nounwind {
; CHECK-LABEL: @combine_redundant_addrspacecast_types(
; CHECK-NEXT: bitcast i32 addrspace(1)* %x to float addrspace(1)*
; CHECK-NEXT: addrspacecast float addrspace(1)* %1 to float*
; CHECK-NEXT: ret
  %y = addrspacecast i32 addrspace(1)* %x to i32 addrspace(3)*
  %z = addrspacecast i32 addrspace(3)* %y to float*
  ret float* %z
}

define <4 x float*> @combine_redundant_addrspacecast_types_vector(<4 x i32 addrspace(1)*> %x) nounwind {
; CHECK-LABEL: @combine_redundant_addrspacecast_types_vector(
; CHECK-NEXT: bitcast <4 x i32 addrspace(1)*> %x to <4 x float addrspace(1)*>
; CHECK-NEXT: addrspacecast <4 x float addrspace(1)*> %1 to <4 x float*>
; CHECK-NEXT: ret
  %y = addrspacecast <4 x i32 addrspace(1)*> %x to <4 x i32 addrspace(3)*>
  %z = addrspacecast <4 x i32 addrspace(3)*> %y to <4 x float*>
  ret <4 x float*> %z
}

define float addrspace(2)* @combine_addrspacecast_bitcast_1(i32 addrspace(1)* %x) nounwind {
; CHECK-LABEL: @combine_addrspacecast_bitcast_1(
; CHECK-NEXT: bitcast i32 addrspace(1)* %x to float addrspace(1)*
; CHECK-NEXT: addrspacecast float addrspace(1)* %1 to float addrspace(2)*
; CHECK-NEXT: ret
  %y = addrspacecast i32 addrspace(1)* %x to i32 addrspace(2)*
  %z = bitcast i32 addrspace(2)* %y to float addrspace(2)*
  ret float addrspace(2)* %z
}

define i32 addrspace(2)* @combine_addrspacecast_bitcast_2(i32 addrspace(1)* %x) nounwind {
; CHECK-LABEL: @combine_addrspacecast_bitcast_2(
; CHECK: addrspacecast i32 addrspace(1)* %x to i32 addrspace(2)*
; CHECK-NEXT: ret
  %y = addrspacecast i32 addrspace(1)* %x to float addrspace(2)*
  %z = bitcast float addrspace(2)* %y to i32 addrspace(2)*
  ret i32 addrspace(2)* %z
}

define i32 addrspace(2)* @combine_bitcast_addrspacecast_1(i32 addrspace(1)* %x) nounwind {
; CHECK-LABEL: @combine_bitcast_addrspacecast_1(
; CHECK: addrspacecast i32 addrspace(1)* %x to i32 addrspace(2)*
; CHECK-NEXT: ret
  %y = bitcast i32 addrspace(1)* %x to i8 addrspace(1)*
  %z = addrspacecast i8 addrspace(1)* %y to i32 addrspace(2)*
  ret i32 addrspace(2)* %z
}

define float addrspace(2)* @combine_bitcast_addrspacecast_2(i32 addrspace(1)* %x) nounwind {
; CHECK-LABEL: @combine_bitcast_addrspacecast_2(
; CHECK: bitcast i32 addrspace(1)* %x to float addrspace(1)*
; CHECK: addrspacecast float addrspace(1)* %1 to float addrspace(2)*
; CHECK-NEXT: ret
  %y = bitcast i32 addrspace(1)* %x to i8 addrspace(1)*
  %z = addrspacecast i8 addrspace(1)* %y to float addrspace(2)*
  ret float addrspace(2)* %z
}

define float addrspace(2)* @combine_addrspacecast_types(i32 addrspace(1)* %x) nounwind {
; CHECK-LABEL: @combine_addrspacecast_types(
; CHECK-NEXT: bitcast i32 addrspace(1)* %x to float addrspace(1)*
; CHECK-NEXT: addrspacecast float addrspace(1)* %1 to float addrspace(2)*
; CHECK-NEXT: ret
  %y = addrspacecast i32 addrspace(1)* %x to float addrspace(2)*
  ret float addrspace(2)* %y
}

define <4 x float addrspace(2)*> @combine_addrspacecast_types_vector(<4 x i32 addrspace(1)*> %x) nounwind {
; CHECK-LABEL: @combine_addrspacecast_types_vector(
; CHECK-NEXT: bitcast <4 x i32 addrspace(1)*> %x to <4 x float addrspace(1)*>
; CHECK-NEXT: addrspacecast <4 x float addrspace(1)*> %1 to <4 x float addrspace(2)*>
; CHECK-NEXT: ret
  %y = addrspacecast <4 x i32 addrspace(1)*> %x to <4 x float addrspace(2)*>
  ret <4 x float addrspace(2)*> %y
}

define i32 @canonicalize_addrspacecast([16 x i32] addrspace(1)* %arr) {
; CHECK-LABEL: @canonicalize_addrspacecast(
; CHECK-NEXT: getelementptr inbounds [16 x i32], [16 x i32] addrspace(1)* %arr, i32 0, i32 0
; CHECK-NEXT: addrspacecast i32 addrspace(1)* %{{[a-zA-Z0-9]+}} to i32*
; CHECK-NEXT: load i32, i32*
; CHECK-NEXT: ret i32
  %p = addrspacecast [16 x i32] addrspace(1)* %arr to i32*
  %v = load i32, i32* %p
  ret i32 %v
}

@const_array = addrspace(2) constant [60 x i8] [i8 2, i8 9, i8 4, i8 22, i8 2, i8 9, i8 4, i8 22, i8 2, i8 9, i8 4, i8 22,
                                                i8 2, i8 9, i8 4, i8 22, i8 2, i8 9, i8 4, i8 22, i8 2, i8 9, i8 4, i8 22,
                                                i8 2, i8 9, i8 4, i8 22, i8 2, i8 9, i8 4, i8 22, i8 2, i8 9, i8 4, i8 22,
                                                i8 2, i8 9, i8 4, i8 22, i8 2, i8 9, i8 4, i8 22, i8 2, i8 9, i8 4, i8 22,
                                                i8 2, i8 9, i8 4, i8 22, i8 2, i8 9, i8 4, i8 22, i8 2, i8 9, i8 4, i8 22 ]

declare void @foo(i8*) nounwind

; A copy from a constant addrspacecast'ed global
; CHECK-LABEL: @memcpy_addrspacecast(
; CHECK-NOT:  call void @llvm.memcpy
define i32 @memcpy_addrspacecast() nounwind {
entry:
  %alloca = alloca i8, i32 48
  call void @llvm.memcpy.p0i8.p1i8.i32(i8* %alloca, i8 addrspace(1)* addrspacecast (i8 addrspace(2)* getelementptr inbounds ([60 x i8], [60 x i8] addrspace(2)* @const_array, i16 0, i16 4) to i8 addrspace(1)*), i32 48, i32 4, i1 false) nounwind
  br label %loop.body

loop.body:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %loop.body ]
  %sum = phi i32 [ 0, %entry ], [ %sum.inc, %loop.body]
  %ptr = getelementptr i8, i8* %alloca, i32 %i
  %load = load i8, i8* %ptr
  %ext = zext i8 %load to i32
  %sum.inc = add i32 %sum, %ext
  %i.inc = add i32 %i, 1
  %cmp = icmp ne i32 %i, 48
  br i1 %cmp, label %loop.body, label %end

end:
  ret i32 %sum.inc
}

; CHECK-LABEL: @constant_fold_null(
; CHECK: i32 addrspace(3)* null to i32 addrspace(4)*
define void @constant_fold_null() #0 {
  %cast = addrspacecast i32 addrspace(3)* null to i32 addrspace(4)*
  store i32 7, i32 addrspace(4)* %cast
  ret void
}

; CHECK-LABEL: @constant_fold_undef(
; CHECK: ret i32 addrspace(4)* undef
define i32 addrspace(4)* @constant_fold_undef() #0 {
  %cast = addrspacecast i32 addrspace(3)* undef to i32 addrspace(4)*
  ret i32 addrspace(4)* %cast
}

; CHECK-LABEL: @constant_fold_null_vector(
; CHECK: addrspacecast (<4 x i32 addrspace(3)*> zeroinitializer to <4 x i32 addrspace(4)*>)
define <4 x i32 addrspace(4)*> @constant_fold_null_vector() #0 {
  %cast = addrspacecast <4 x i32 addrspace(3)*> zeroinitializer to <4 x i32 addrspace(4)*>
  ret <4 x i32 addrspace(4)*> %cast
}

; CHECK-LABEL: @constant_fold_inttoptr(
; CHECK: addrspacecast (i32 addrspace(3)* inttoptr (i32 -1 to i32 addrspace(3)*) to i32 addrspace(4)*)
define void @constant_fold_inttoptr() #0 {
  %cast = addrspacecast i32 addrspace(3)* inttoptr (i32 -1 to i32 addrspace(3)*) to i32 addrspace(4)*
  store i32 7, i32 addrspace(4)* %cast
  ret void
}

; CHECK-LABEL: @constant_fold_gep_inttoptr(
; CHECK: addrspacecast (i32 addrspace(3)* inttoptr (i64 1274 to i32 addrspace(3)*) to i32 addrspace(4)*)
define void @constant_fold_gep_inttoptr() #0 {
  %k = inttoptr i32 1234 to i32 addrspace(3)*
  %gep = getelementptr i32, i32 addrspace(3)* %k, i32 10
  %cast = addrspacecast i32 addrspace(3)* %gep to i32 addrspace(4)*
  store i32 7, i32 addrspace(4)* %cast
  ret void
}

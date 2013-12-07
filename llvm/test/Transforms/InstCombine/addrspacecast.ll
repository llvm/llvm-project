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
; CHECK: addrspacecast i32 addrspace(1)* %x to float*
; CHECK-NEXT: ret
  %y = addrspacecast i32 addrspace(1)* %x to i32 addrspace(3)*
  %z = addrspacecast i32 addrspace(3)* %y to float*
  ret float* %z
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
  call void @llvm.memcpy.p0i8.p1i8.i32(i8* %alloca, i8 addrspace(1)* addrspacecast (i8 addrspace(2)* getelementptr inbounds ([60 x i8] addrspace(2)* @const_array, i16 0, i16 4) to i8 addrspace(1)*), i32 48, i32 4, i1 false) nounwind
  br label %loop.body

loop.body:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %loop.body ]
  %sum = phi i32 [ 0, %entry ], [ %sum.inc, %loop.body]
  %ptr = getelementptr i8* %alloca, i32 %i
  %load = load i8* %ptr
  %ext = zext i8 %load to i32
  %sum.inc = add i32 %sum, %ext
  %i.inc = add i32 %i, 1
  %cmp = icmp ne i32 %i, 48
  br i1 %cmp, label %loop.body, label %end

end:
  ret i32 %sum.inc
}


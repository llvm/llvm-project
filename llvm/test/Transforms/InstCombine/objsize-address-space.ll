; RUN: opt -S -passes=instcombine -o - %s | FileCheck %s
target datalayout = "e-p:32:32:32-p1:64:64:64-p2:8:8:8-p3:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32"

declare i32 @llvm.objectsize.i32.p0i8(i8*, i1) nounwind readonly
declare i32 @llvm.objectsize.i32.p1i8(i8 addrspace(1)*, i1) nounwind readonly
declare i32 @llvm.objectsize.i32.p2i8(i8 addrspace(2)*, i1) nounwind readonly
declare i32 @llvm.objectsize.i32.p3i8(i8 addrspace(3)*, i1) nounwind readonly
declare i16 @llvm.objectsize.i16.p3i8(i8 addrspace(3)*, i1) nounwind readonly

@array_as2 = private addrspace(2) global [60 x i8] zeroinitializer, align 4

@array_as1_pointers = private global [10 x i32 addrspace(1)*] zeroinitializer, align 4
@array_as2_pointers = private global [24 x i32 addrspace(2)*] zeroinitializer, align 4
@array_as3_pointers = private global [42 x i32 addrspace(3)*] zeroinitializer, align 4

@array_as2_as1_pointer_pointers = private global [16 x i32 addrspace(2)* addrspace(1)*] zeroinitializer, align 4


@a_as3 = private addrspace(3) global [60 x i8] zeroinitializer, align 1

define i32 @foo_as3() nounwind {
; CHECK-LABEL: @foo_as3(
; CHECK-NEXT: ret i32 60
  %1 = call i32 @llvm.objectsize.i32.p3i8(i8 addrspace(3)* getelementptr inbounds ([60 x i8], [60 x i8] addrspace(3)* @a_as3, i32 0, i32 0), i1 false)
  ret i32 %1
}

define i16 @foo_as3_i16() nounwind {
; CHECK-LABEL: @foo_as3_i16(
; CHECK-NEXT: ret i16 60
  %1 = call i16 @llvm.objectsize.i16.p3i8(i8 addrspace(3)* getelementptr inbounds ([60 x i8], [60 x i8] addrspace(3)* @a_as3, i32 0, i32 0), i1 false)
  ret i16 %1
}

@a_alias = weak alias [60 x i8], [60 x i8] addrspace(3)* @a_as3
define i32 @foo_alias() nounwind {
  %1 = call i32 @llvm.objectsize.i32.p3i8(i8 addrspace(3)* getelementptr inbounds ([60 x i8], [60 x i8] addrspace(3)* @a_alias, i32 0, i32 0), i1 false)
  ret i32 %1
}

define i32 @array_as2_size() {
; CHECK-LABEL: @array_as2_size(
; CHECK-NEXT: ret i32 60
  %bc = bitcast [60 x i8] addrspace(2)* @array_as2 to i8 addrspace(2)*
  %1 = call i32 @llvm.objectsize.i32.p2i8(i8 addrspace(2)* %bc, i1 false)
  ret i32 %1
}

define i32 @pointer_array_as1() {
; CHECK-LABEL: @pointer_array_as1(
; CHECK-NEXT: ret i32 80
  %bc = addrspacecast [10 x i32 addrspace(1)*]* @array_as1_pointers to i8 addrspace(1)*
  %1 = call i32 @llvm.objectsize.i32.p1i8(i8 addrspace(1)* %bc, i1 false)
  ret i32 %1
}

define i32 @pointer_array_as2() {
; CHECK-LABEL: @pointer_array_as2(
; CHECK-NEXT: ret i32 24
  %bc = bitcast [24 x i32 addrspace(2)*]* @array_as2_pointers to i8*
  %1 = call i32 @llvm.objectsize.i32.p0i8(i8* %bc, i1 false)
  ret i32 %1
}

define i32 @pointer_array_as3() {
; CHECK-LABEL: @pointer_array_as3(
; CHECK-NEXT: ret i32 84
  %bc = bitcast [42 x i32 addrspace(3)*]* @array_as3_pointers to i8*
  %1 = call i32 @llvm.objectsize.i32.p0i8(i8* %bc, i1 false)
  ret i32 %1
}

define i32 @pointer_pointer_array_as2_as1() {
; CHECK-LABEL: @pointer_pointer_array_as2_as1(
; CHECK-NEXT: ret i32 128
  %bc = bitcast [16 x i32 addrspace(2)* addrspace(1)*]* @array_as2_as1_pointer_pointers to i8*
  %1 = call i32 @llvm.objectsize.i32.p0i8(i8* %bc, i1 false)
  ret i32 %1
}


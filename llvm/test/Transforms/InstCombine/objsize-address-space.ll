; RUN: opt -S -passes=instcombine -o - %s | FileCheck %s
target datalayout = "e-p:32:32:32-p1:64:64:64-p2:8:8:8-p3:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32"

declare i32 @llvm.objectsize.i32.p0(ptr, i1) nounwind readonly
declare i32 @llvm.objectsize.i32.p1(ptr addrspace(1), i1) nounwind readonly
declare i32 @llvm.objectsize.i32.p2(ptr addrspace(2), i1) nounwind readonly
declare i32 @llvm.objectsize.i32.p3(ptr addrspace(3), i1) nounwind readonly
declare i16 @llvm.objectsize.i16.p3(ptr addrspace(3), i1) nounwind readonly

@array_as2 = private addrspace(2) global [60 x i8] zeroinitializer, align 4

@array_as1_pointers = private global [10 x ptr addrspace(1)] zeroinitializer, align 4
@array_as2_pointers = private global [24 x ptr addrspace(2)] zeroinitializer, align 4
@array_as3_pointers = private global [42 x ptr addrspace(3)] zeroinitializer, align 4

@array_as2_as1_pointer_pointers = private global [16 x ptr addrspace(1)] zeroinitializer, align 4


@a_as3 = private addrspace(3) global [60 x i8] zeroinitializer, align 1

define i32 @foo_as3() nounwind {
; CHECK-LABEL: @foo_as3(
; CHECK-NEXT: ret i32 60
  %1 = call i32 @llvm.objectsize.i32.p3(ptr addrspace(3) @a_as3, i1 false)
  ret i32 %1
}

define i16 @foo_as3_i16() nounwind {
; CHECK-LABEL: @foo_as3_i16(
; CHECK-NEXT: ret i16 60
  %1 = call i16 @llvm.objectsize.i16.p3(ptr addrspace(3) @a_as3, i1 false)
  ret i16 %1
}

@a_alias = weak alias [60 x i8], ptr addrspace(3) @a_as3
define i32 @foo_alias() nounwind {
  %1 = call i32 @llvm.objectsize.i32.p3(ptr addrspace(3) @a_alias, i1 false)
  ret i32 %1
}

define i32 @array_as2_size() {
; CHECK-LABEL: @array_as2_size(
; CHECK-NEXT: ret i32 60
  %1 = call i32 @llvm.objectsize.i32.p2(ptr addrspace(2) @array_as2, i1 false)
  ret i32 %1
}

define i32 @pointer_array_as1() {
; CHECK-LABEL: @pointer_array_as1(
; CHECK-NEXT: ret i32 80
  %bc = addrspacecast ptr @array_as1_pointers to ptr addrspace(1)
  %1 = call i32 @llvm.objectsize.i32.p1(ptr addrspace(1) %bc, i1 false)
  ret i32 %1
}

define i32 @pointer_array_as2() {
; CHECK-LABEL: @pointer_array_as2(
; CHECK-NEXT: ret i32 24
  %1 = call i32 @llvm.objectsize.i32.p0(ptr @array_as2_pointers, i1 false)
  ret i32 %1
}

define i32 @pointer_array_as3() {
; CHECK-LABEL: @pointer_array_as3(
; CHECK-NEXT: ret i32 84
  %1 = call i32 @llvm.objectsize.i32.p0(ptr @array_as3_pointers, i1 false)
  ret i32 %1
}

define i32 @pointer_pointer_array_as2_as1() {
; CHECK-LABEL: @pointer_pointer_array_as2_as1(
; CHECK-NEXT: ret i32 128
  %1 = call i32 @llvm.objectsize.i32.p0(ptr @array_as2_as1_pointer_pointers, i1 false)
  ret i32 %1
}


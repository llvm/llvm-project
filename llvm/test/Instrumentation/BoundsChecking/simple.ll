; RUN: opt < %s -bounds-checking -S | FileCheck %s
; RUN: opt < %s -passes=bounds-checking -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@.str = private constant [8 x i8] c"abcdefg\00"   ; <[8 x i8]*>

@.str_as1 = private addrspace(1) constant [8 x i8] c"abcdefg\00"   ; <[8 x i8] addrspace(1)*>


declare noalias i8* @malloc(i64) nounwind
declare noalias i8* @calloc(i64, i64) nounwind
declare noalias i8* @realloc(i8* nocapture, i64) nounwind

; CHECK: @f1
define void @f1() nounwind {
  %1 = tail call i8* @malloc(i64 32)
  %2 = bitcast i8* %1 to i32*
  %idx = getelementptr inbounds i32, i32* %2, i64 2
; CHECK-NOT: trap
  store i32 3, i32* %idx, align 4
  ret void
}

; CHECK: @f2
define void @f2() nounwind {
  %1 = tail call i8* @malloc(i64 32)
  %2 = bitcast i8* %1 to i32*
  %idx = getelementptr inbounds i32, i32* %2, i64 8
; CHECK: trap
  store i32 3, i32* %idx, align 4
  ret void
}

; CHECK: @f3
define void @f3(i64 %x) nounwind {
  %1 = tail call i8* @calloc(i64 4, i64 %x)
  %2 = bitcast i8* %1 to i32*
  %idx = getelementptr inbounds i32, i32* %2, i64 8
; CHECK: mul i64 4, %
; CHECK: sub i64 {{.*}}, 32
; CHECK-NEXT: icmp ult i64 {{.*}}, 32
; CHECK-NEXT: icmp ult i64 {{.*}}, 4
; CHECK-NEXT: or i1
; CHECK: trap
  store i32 3, i32* %idx, align 4
  ret void
}

; CHECK: @store_volatile
define void @store_volatile(i64 %x) nounwind {
  %1 = tail call i8* @calloc(i64 4, i64 %x)
  %2 = bitcast i8* %1 to i32*
  %idx = getelementptr inbounds i32, i32* %2, i64 8
; CHECK-NOT: trap
  store volatile i32 3, i32* %idx, align 4
  ret void
}

; CHECK: @f4
define void @f4(i64 %x) nounwind {
  %1 = tail call i8* @realloc(i8* null, i64 %x) nounwind
  %2 = bitcast i8* %1 to i32*
  %idx = getelementptr inbounds i32, i32* %2, i64 8
; CHECK: trap
  %3 = load i32, i32* %idx, align 4
  ret void
}

; CHECK: @f5
define void @f5(i64 %x) nounwind {
  %idx = getelementptr inbounds [8 x i8], [8 x i8]* @.str, i64 0, i64 %x
; CHECK: trap
  %1 = load i8, i8* %idx, align 4
  ret void
}

define void @f5_as1(i64 %x) nounwind {
; CHECK: @f5_as1
  %idx = getelementptr inbounds [8 x i8], [8 x i8] addrspace(1)* @.str_as1, i64 0, i64 %x
  ; CHECK: sub i16
  ; CHECK: icmp ult i16
; CHECK: trap
  %1 = load i8, i8 addrspace(1)* %idx, align 4
  ret void
}

; CHECK: @f6
define void @f6(i64 %x) nounwind {
  %1 = alloca i128
; CHECK-NOT: trap
  %2 = load i128, i128* %1, align 4
  ret void
}

; CHECK: @f7
define void @f7(i64 %x) nounwind {
  %1 = alloca i128, i64 %x
; CHECK: mul i64 16,
; CHECK: trap
  %2 = load i128, i128* %1, align 4
  ret void
}

; CHECK: @f8
define void @f8() nounwind {
  %1 = alloca i128
  %2 = alloca i128
  %3 = select i1 undef, i128* %1, i128* %2
; CHECK-NOT: trap
  %4 = load i128, i128* %3, align 4
  ret void
}

; CHECK: @f9
define void @f9(i128* %arg) nounwind {
  %1 = alloca i128
  %2 = select i1 undef, i128* %arg, i128* %1
; CHECK-NOT: trap
  %3 = load i128, i128* %2, align 4
  ret void
}

; CHECK: @f10
define void @f10(i64 %x, i64 %y) nounwind {
  %1 = alloca i128, i64 %x
  %2 = alloca i128, i64 %y
  %3 = select i1 undef, i128* %1, i128* %2
; CHECK: select
; CHECK: select
; CHECK: trap
  %4 = load i128, i128* %3, align 4
  ret void
}

; CHECK: @f11
define void @f11(i128* byval(i128) %x) nounwind {
  %1 = bitcast i128* %x to i8*
  %2 = getelementptr inbounds i8, i8* %1, i64 16
; CHECK: br label
  %3 = load i8, i8* %2, align 4
  ret void
}

; CHECK: @f11_as1
define void @f11_as1(i128 addrspace(1)* byval(i128) %x) nounwind {
  %1 = bitcast i128 addrspace(1)* %x to i8 addrspace(1)*
  %2 = getelementptr inbounds i8, i8 addrspace(1)* %1, i16 16
; CHECK: br label
  %3 = load i8, i8 addrspace(1)* %2, align 4
  ret void
}

; CHECK: @f12
define i64 @f12(i64 %x, i64 %y) nounwind {
  %1 = tail call i8* @calloc(i64 1, i64 %x)
; CHECK: mul i64 %y, 8
; CHECK: trap
  %2 = bitcast i8* %1 to i64*
  %3 = getelementptr inbounds i64, i64* %2, i64 %y
  %4 = load i64, i64* %3, align 8
  ret i64 %4
}

; CHECK: @load_volatile
define i64 @load_volatile(i64 %x, i64 %y) nounwind {
  %1 = tail call i8* @calloc(i64 1, i64 %x)
; CHECK-NOT: trap
  %2 = bitcast i8* %1 to i64*
  %3 = getelementptr inbounds i64, i64* %2, i64 %y
  %4 = load volatile i64, i64* %3, align 8
  ret i64 %4
}

; PR17402
; CHECK-LABEL: @f13
define void @f13() nounwind {
entry:
  br label %alive

dead:
  ; Self-refential GEPs can occur in dead code.
  %incdec.ptr = getelementptr inbounds i32, i32* %incdec.ptr, i64 1
  ; CHECK: %incdec.ptr = getelementptr inbounds i32, i32* %incdec.ptr
  %l = load i32, i32* %incdec.ptr
  br label %alive

alive:
  ret void
}

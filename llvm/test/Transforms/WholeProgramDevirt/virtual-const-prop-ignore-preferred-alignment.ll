;; Demonstrate that the ABI alignment is used over the preferred alignment.
;;
;; In both runs, pointers are 32-bit but we can only store the function returing
;; the 64-bit constant in the vtable if the ABI alignment for an i64 is 32 since
;; we cannot guarantee a 64-bit ABI alignment if the vtable is 32-bit aligned.
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility --data-layout="e-p:32:32-i64:32:64" %s | FileCheck %s --check-prefixes=COMMON,ABI32
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility --data-layout="e-p:32:32-i64:64:64" %s | FileCheck %s --check-prefixes=COMMON,ABI64

; ABI32:      [[VT6DATA:@[^ ]*]] = {{.*}} { [4 x i8], [2 x ptr], [8 x i8] } 
; ABI32-SAME:   [8 x i8] c"\05\00\00\00\00\00\00\00"
; ABI64:      [[VT6DATA:@[^ ]*]] = {{.*}} { [4 x i8], [2 x ptr], [0 x i8] } 
; ABI64-SAME:   zeroinitializer
; COMMON-SAME: }, !type [[T:![0-9]+]]
@vt6 = constant [2 x ptr] [
ptr @vf10i8,
ptr @vf5i64
], !type !1

; ABI32:      [[VT6DATA:@[^ ]*]] = {{.*}} { [4 x i8], [2 x ptr], [8 x i8] } 
; ABI32-SAME:   [8 x i8] c"\06\00\00\00\00\00\00\00"
; ABI64:      [[VT6DATA:@[^ ]*]] = {{.*}} { [4 x i8], [2 x ptr], [0 x i8] } 
; ABI64-SAME:   zeroinitializer
; COMMON-SAME: }, !type [[T]]
@vt7 = constant [2 x ptr] [
ptr @vf9i8,
ptr @vf6i64
], !type !1

define i1 @vf0i1(ptr %this) readnone {
  ret i1 0
}

define i1 @vf1i1(ptr %this) readnone {
  ret i1 1
}

define i8 @vf0i8(ptr %this) readnone {
  ret i8 2
}

define i8 @vf1i8(ptr %this) readnone {
  ret i8 3
}

define i32 @vf1i32(ptr %this) readnone {
  ret i32 1
}

define i32 @vf2i32(ptr %this) readnone {
  ret i32 2
}

define i32 @vf3i32(ptr %this) readnone {
  ret i32 3
}

define i32 @vf4i32(ptr %this) readnone {
  ret i32 4
}

define i64 @vf5i64(ptr %this) readnone {
  ret i64 5
}

define i64 @vf6i64(ptr %this) readnone {
  ret i64 6
}

define i8 @vf9i8(ptr %this) readnone {
  ret i8 10
}

define i8 @vf10i8(ptr %this) readnone {
  ret i8 11
}

; COMMON-LABEL: define i8 @call0(
define i8 @call0(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x ptr], ptr %vtable, i32 0, i32 0
  %fptr = load ptr, ptr %fptrptr
  %result = call i8 %fptr(ptr %obj)
  ret i8 %result
  ; COMMON: [[VTGEP:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 -1
  ; COMMON: [[VTLOAD:%[^ ]*]] = load i8, ptr [[VTGEP]]
  ; COMMON: ret i8 [[VTLOAD]]
}

; COMMON-LABEL: define i64 @call1(
define i64 @call1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x ptr], ptr %vtable, i32 0, i32 1
  %fptr = load ptr, ptr %fptrptr
  %result = call i64 %fptr(ptr %obj)
  ret i64 %result
  ; ABI32: [[VTGEP:%[^ ]*]] = getelementptr i8, ptr %vtable, i32 8
  ; ABI32-NEXT: [[VTLOAD:%[^ ]*]] = load i64, ptr [[VTGEP]]
  ; ABI64: [[VTGEP:%[^ ]*]] = getelementptr [3 x ptr], ptr %vtable, i32 0, i32 1
  ; ABI64-NEXT: [[FUNC:%[^ ]*]] = load ptr, ptr [[VTGEP]], align 4
  ; ABI64-NEXT: [[VTLOAD:%[^ ]*]] = call i64 [[FUNC]](ptr %obj)
  ; COMMON-NEXT: ret i64 [[VTLOAD]]
}

; COMMON: [[T]] = !{i32 4, !"typeid"}

!1 = !{i32 0, !"typeid"}

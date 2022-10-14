; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: private constant { [8 x i8], [1 x ptr], [0 x i8] } { [8 x i8] c"\00\00\00\00\00\00\00\01", [1 x ptr] [ptr @vf1], [0 x i8] zeroinitializer }, !type [[T8:![0-9]+]]
; CHECK: private constant { [8 x i8], [1 x ptr], [0 x i8] } { [8 x i8] c"\00\00\00\00\00\00\00\02", [1 x ptr] [ptr @vf2], [0 x i8] zeroinitializer }, !type [[T8]]
; CHECK: private constant { [8 x i8], [1 x ptr], [0 x i8] } { [8 x i8] c"\00\00\00\00\00\00\00\01", [1 x ptr] [ptr @vf4], [0 x i8] zeroinitializer }, !type [[T8]]
; CHECK: private constant { [8 x i8], [1 x ptr], [0 x i8] } { [8 x i8] c"\00\00\00\00\00\00\00\02", [1 x ptr] [ptr @vf8], [0 x i8] zeroinitializer }, !type [[T8]]

@vt1 = constant [1 x ptr] [ptr @vf1], !type !0
@vt2 = constant [1 x ptr] [ptr @vf2], !type !0
@vt4 = constant [1 x ptr] [ptr @vf4], !type !0
@vt8 = constant [1 x ptr] [ptr @vf8], !type !0

define i1 @vf1(ptr %this, i32 %arg) readnone {
  %and = and i32 %arg, 1
  %cmp = icmp ne i32 %and, 0
  ret i1 %cmp
}

define i1 @vf2(ptr %this, i32 %arg) readnone {
  %and = and i32 %arg, 2
  %cmp = icmp ne i32 %and, 0
  ret i1 %cmp
}

define i1 @vf4(ptr %this, i32 %arg) readnone {
  %and = and i32 %arg, 4
  %cmp = icmp ne i32 %and, 0
  ret i1 %cmp
}

define i1 @vf8(ptr %this, i32 %arg) readnone {
  %and = and i32 %arg, 8
  %cmp = icmp ne i32 %and, 0
  ret i1 %cmp
}

; CHECK: define i1 @call1
define i1 @call1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: getelementptr {{.*}} -1
  ; CHECK: and {{.*}}, 1
  %result = call i1 %fptr(ptr %obj, i32 5)
  ret i1 %result
}

; CHECK: define i1 @call2
define i1 @call2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: getelementptr {{.*}} -1
  ; CHECK: and {{.*}}, 2
  %result = call i1 %fptr(ptr %obj, i32 10)
  ret i1 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

; CHECK: [[T8]] = !{i32 8, !"typeid"}
!0 = !{i32 0, !"typeid"}

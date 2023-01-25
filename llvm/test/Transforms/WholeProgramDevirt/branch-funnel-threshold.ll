; RUN: opt -passes=wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -wholeprogramdevirt-branch-funnel-threshold=1 -S -o - %s | not grep @llvm.icall.branch.funnel | count 0

; RUN: opt -passes=wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -wholeprogramdevirt-branch-funnel-threshold=10 -S -o - %s | grep @llvm.icall.branch.funnel | count 4

; RUN: opt -passes=wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -wholeprogramdevirt-branch-funnel-threshold=100 -S -o - %s | grep @llvm.icall.branch.funnel | count 5

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1_1 = constant [1 x ptr] [ptr @vf1_1], !type !0
@vt1_2 = constant [1 x ptr] [ptr @vf1_2], !type !0

declare i32 @vf1_1(ptr %this, i32 %arg)
declare i32 @vf1_2(ptr %this, i32 %arg)

@vt2_1 = constant [1 x ptr] [ptr @vf2_1], !type !1
@vt2_2 = constant [1 x ptr] [ptr @vf2_2], !type !1
@vt2_3 = constant [1 x ptr] [ptr @vf2_3], !type !1
@vt2_4 = constant [1 x ptr] [ptr @vf2_4], !type !1
@vt2_5 = constant [1 x ptr] [ptr @vf2_5], !type !1
@vt2_6 = constant [1 x ptr] [ptr @vf2_6], !type !1
@vt2_7 = constant [1 x ptr] [ptr @vf2_7], !type !1
@vt2_8 = constant [1 x ptr] [ptr @vf2_8], !type !1
@vt2_9 = constant [1 x ptr] [ptr @vf2_9], !type !1
@vt2_10 = constant [1 x ptr] [ptr @vf2_10], !type !1
@vt2_11 = constant [1 x ptr] [ptr @vf2_11], !type !1

declare i32 @vf2_1(ptr %this, i32 %arg)
declare i32 @vf2_2(ptr %this, i32 %arg)
declare i32 @vf2_3(ptr %this, i32 %arg)
declare i32 @vf2_4(ptr %this, i32 %arg)
declare i32 @vf2_5(ptr %this, i32 %arg)
declare i32 @vf2_6(ptr %this, i32 %arg)
declare i32 @vf2_7(ptr %this, i32 %arg)
declare i32 @vf2_8(ptr %this, i32 %arg)
declare i32 @vf2_9(ptr %this, i32 %arg)
declare i32 @vf2_10(ptr %this, i32 %arg)
declare i32 @vf2_11(ptr %this, i32 %arg)

@vt3_1 = constant [1 x ptr] [ptr @vf3_1], !type !2
@vt3_2 = constant [1 x ptr] [ptr @vf3_2], !type !2

declare i32 @vf3_1(ptr %this, i32 %arg)
declare i32 @vf3_2(ptr %this, i32 %arg)

@vt4_1 = constant [1 x ptr] [ptr @vf4_1], !type !3
@vt4_2 = constant [1 x ptr] [ptr @vf4_2], !type !3

declare i32 @vf4_1(ptr %this, i32 %arg)
declare i32 @vf4_2(ptr %this, i32 %arg)

define i32 @fn1(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

define i32 @fn2(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

define i32 @fn3(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !4)
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}
!3 = !{i32 0, !4}
!4 = distinct !{}

attributes #0 = { "target-features"="+retpoline" }

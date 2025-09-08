; A variant of branch-funnel.ll where we just check that the funnels' entry counts
; are correctly set.
;
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck --check-prefixes=RETP %s
; RUN: sed -e 's,+retpoline,-retpoline,g' %s | opt -S -passes=wholeprogramdevirt -whole-program-visibility | FileCheck --check-prefixes=NORETP %s
; RUN: opt -passes=wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -S -o - %s | FileCheck --check-prefixes=RETP %s
; RUN: opt -passes='wholeprogramdevirt,default<O3>' -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t  -S -o - %s | FileCheck --check-prefixes=O3 %s

; RETP: define hidden void @__typeid_typeid1_0_branch_funnel(ptr nest %0, ...) !prof !11
; RETP: define hidden void @__typeid_typeid1_rv_0_branch_funnel(ptr nest %0, ...) !prof !11
; RETP: define internal void @branch_funnel(ptr nest %0, ...) !prof !10
; RETP: define internal void @branch_funnel.1(ptr nest %0, ...) !prof !10 
; RETP: !10 = !{!"function_entry_count", i64 1000}
; RETP: !11 = !{!"function_entry_count", i64 3000}

; NORETP: define hidden void @__typeid_typeid1_0_branch_funnel(ptr nest %0, ...) !prof !11
; NORETP: define hidden void @__typeid_typeid1_rv_0_branch_funnel(ptr nest %0, ...) !prof !11
; NORETP: define internal void @branch_funnel(ptr nest %0, ...) !prof !11
; NORETP: define internal void @branch_funnel.1(ptr nest %0, ...) !prof !11
; NORETP: !11 = !{!"unknown"}

; O3: define hidden void @__typeid_typeid1_0_branch_funnel(ptr nest %0, ...) local_unnamed_addr #5 !prof !11
; O3: define hidden void @__typeid_typeid1_rv_0_branch_funnel(ptr nest %0, ...) local_unnamed_addr #5 !prof !11
; O3: define internal void @branch_funnel(ptr nest %0, ...) unnamed_addr #5 !prof !10
; O3: define internal void @branch_funnel.1(ptr nest %0, ...) unnamed_addr #5 !prof !10
; O3: define hidden void @__typeid_typeid3_0_branch_funnel(ptr nest %0, ...) local_unnamed_addr #5 !prof !12
; O3: define hidden void @__typeid_typeid3_rv_0_branch_funnel(ptr nest %0, ...) local_unnamed_addr #5 !prof !12
; O3: !10 = !{!"function_entry_count", i64 1000}
; O3: !11 = !{!"function_entry_count", i64 3000}
; O3: !12 = !{!"unknown"}

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

declare ptr @llvm.load.relative.i32(ptr, i32)

;; These are relative vtables equivalent to the ones above.
@vt1_1_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf1_1 to i64), i64 ptrtoint (ptr @vt1_1_rv to i64)) to i32)], !type !5
@vt1_2_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf1_2 to i64), i64 ptrtoint (ptr @vt1_2_rv to i64)) to i32)], !type !5

@vt2_1_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_1 to i64), i64 ptrtoint (ptr @vt2_1_rv to i64)) to i32)], !type !6
@vt2_2_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_2 to i64), i64 ptrtoint (ptr @vt2_2_rv to i64)) to i32)], !type !6
@vt2_3_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_3 to i64), i64 ptrtoint (ptr @vt2_3_rv to i64)) to i32)], !type !6
@vt2_4_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_4 to i64), i64 ptrtoint (ptr @vt2_4_rv to i64)) to i32)], !type !6
@vt2_5_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_5 to i64), i64 ptrtoint (ptr @vt2_5_rv to i64)) to i32)], !type !6
@vt2_6_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_6 to i64), i64 ptrtoint (ptr @vt2_6_rv to i64)) to i32)], !type !6
@vt2_7_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_7 to i64), i64 ptrtoint (ptr @vt2_7_rv to i64)) to i32)], !type !6
@vt2_8_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_8 to i64), i64 ptrtoint (ptr @vt2_8_rv to i64)) to i32)], !type !6
@vt2_9_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_9 to i64), i64 ptrtoint (ptr @vt2_9_rv to i64)) to i32)], !type !6
@vt2_10_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_10 to i64), i64 ptrtoint (ptr @vt2_10_rv to i64)) to i32)], !type !6
@vt2_11_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_11 to i64), i64 ptrtoint (ptr @vt2_11_rv to i64)) to i32)], !type !6

@vt3_1_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf3_1 to i64), i64 ptrtoint (ptr @vt3_1_rv to i64)) to i32)], !type !7
@vt3_2_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf3_2 to i64), i64 ptrtoint (ptr @vt3_2_rv to i64)) to i32)], !type !7

@vt4_1_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf4_1 to i64), i64 ptrtoint (ptr @vt4_1_rv to i64)) to i32)], !type !8
@vt4_2_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf4_2 to i64), i64 ptrtoint (ptr @vt4_2_rv to i64)) to i32)], !type !8


define i32 @fn1(ptr %obj) #0 !prof !10 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

define i32 @fn1_rv(ptr %obj) #0 !prof !10 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1_rv")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0)
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

define i32 @fn2(ptr %obj) #0 !prof !10 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

define i32 @fn2_rv(ptr %obj) #0 !prof !10 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2_rv")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0)
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

define i32 @fn3(ptr %obj) #0 !prof !10 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !4)
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

define i32 @fn3_rv(ptr %obj) #0 !prof !10 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !9)
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0)
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

define i32 @fn4(ptr %obj) #0 !prof !10 {
  %p = call i1 @llvm.type.test(ptr @vt1_1, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr @vt1_1
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

define i32 @fn4_cpy(ptr %obj) #0 !prof !10 {
  %p = call i1 @llvm.type.test(ptr @vt1_1, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr @vt1_1
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

define i32 @fn4_rv(ptr %obj) #0 !prof !10 {
  %p = call i1 @llvm.type.test(ptr @vt1_1_rv, metadata !"typeid1_rv")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr @vt1_1_rv, i32 0)
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

define i32 @fn4_rv_cpy(ptr %obj) #0 !prof !10 {
  %p = call i1 @llvm.type.test(ptr @vt1_1_rv, metadata !"typeid1_rv")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr @vt1_1_rv, i32 0)
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
!5 = !{i32 0, !"typeid1_rv"}
!6 = !{i32 0, !"typeid2_rv"}
!7 = !{i32 0, !"typeid3_rv"}
!8 = !{i32 0, !9}
!9 = distinct !{}
!10 = !{!"function_entry_count", i64 1000}

attributes #0 = { "target-features"="+retpoline" }

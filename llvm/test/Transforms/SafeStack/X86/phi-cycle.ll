; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

%struct.small = type { i8 }

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Address-of a structure taken in a function with a loop where
; the alloca is an incoming value to a PHI node and a use of that PHI
; node is also an incoming value.
; Verify that the address-of analysis does not get stuck in infinite
; recursion when chasing the alloca through the PHI nodes.
; Requires protector.
define i32 @foo(i32 %arg) nounwind uwtable safestack {
bb:
  ; CHECK: __safestack_unsafe_stack_ptr
  %tmp = alloca ptr, align 8
  %tmp1 = call i32 (...) @dummy(ptr %tmp) nounwind
  %tmp2 = load ptr, ptr %tmp, align 8
  %tmp3 = ptrtoint ptr %tmp2 to i64
  %tmp4 = trunc i64 %tmp3 to i32
  %tmp5 = icmp sgt i32 %tmp4, 0
  br i1 %tmp5, label %bb6, label %bb21

bb6:                                              ; preds = %bb17, %bb
  %tmp7 = phi ptr [ %tmp19, %bb17 ], [ %tmp2, %bb ]
  %tmp8 = phi i64 [ %tmp20, %bb17 ], [ 1, %bb ]
  %tmp9 = phi i32 [ %tmp14, %bb17 ], [ %tmp1, %bb ]
  %tmp11 = load i8, ptr %tmp7, align 1
  %tmp12 = icmp eq i8 %tmp11, 1
  %tmp13 = add nsw i32 %tmp9, 8
  %tmp14 = select i1 %tmp12, i32 %tmp13, i32 %tmp9
  %tmp15 = trunc i64 %tmp8 to i32
  %tmp16 = icmp eq i32 %tmp15, %tmp4
  br i1 %tmp16, label %bb21, label %bb17

bb17:                                             ; preds = %bb6
  %tmp18 = getelementptr inbounds ptr, ptr %tmp, i64 %tmp8
  %tmp19 = load ptr, ptr %tmp18, align 8
  %tmp20 = add i64 %tmp8, 1
  br label %bb6

bb21:                                             ; preds = %bb6, %bb
  %tmp22 = phi i32 [ %tmp1, %bb ], [ %tmp14, %bb6 ]
  %tmp23 = call i32 (...) @dummy(i32 %tmp22) nounwind
  ret i32 undef
}

declare i32 @dummy(...)

; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { [100 x i32] }

; Safe access to a byval argument.
define i32 @ByValSafe(ptr byval(%struct.S) nocapture readonly align 8 %zzz) norecurse nounwind readonly safestack uwtable {
entry:
  ; CHECK-LABEL: @ByValSafe
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  ; CHECK: ret i32
  %arrayidx = getelementptr inbounds %struct.S, ptr %zzz, i64 0, i32 0, i64 3
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

; Unsafe access to a byval argument.
; Argument is copied to the unsafe stack.
define i32 @ByValUnsafe(ptr byval(%struct.S) nocapture readonly align 8 %zzz, i64 %idx) norecurse nounwind readonly safestack uwtable {
entry:
  ; CHECK-LABEL: @ByValUnsafe
  ; CHECK: %[[A:.*]] = load {{.*}} @__safestack_unsafe_stack_ptr
  ; CHECK: store {{.*}} @__safestack_unsafe_stack_ptr
  ; CHECK: %[[B:.*]] = getelementptr i8, ptr %[[A]], i32 -400
  ; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[B]], ptr align 8 %zzz, i64 400, i1 false)
  ; CHECK: ret i32
  %arrayidx = getelementptr inbounds %struct.S, ptr %zzz, i64 0, i32 0, i64 %idx
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

; Unsafe access to a byval argument.
; Argument is copied to the unsafe stack.
; Check that dest align of memcpy is set according to datalayout prefered alignment
define i32 @ByValUnsafe2(ptr byval(%struct.S) nocapture readonly %zzz, i64 %idx) norecurse nounwind readonly safestack uwtable {
entry:
  ; CHECK-LABEL: @ByValUnsafe
  ; CHECK: %[[A:.*]] = load {{.*}} @__safestack_unsafe_stack_ptr
  ; CHECK: store {{.*}} @__safestack_unsafe_stack_ptr
  ; CHECK: %[[B:.*]] = getelementptr i8, ptr %[[A]], i32 -400
  ; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[B]], ptr %zzz, i64 400, i1 false)
  ; CHECK: ret i32
  %arrayidx = getelementptr inbounds %struct.S, ptr %zzz, i64 0, i32 0, i64 %idx
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

; Highly aligned byval argument.
define i32 @ByValUnsafeAligned(ptr byval(%struct.S) nocapture readonly align 64 %zzz, i64 %idx) norecurse nounwind readonly safestack uwtable {
entry:
  ; CHECK-LABEL: @ByValUnsafeAligned
  ; CHECK: %[[A:.*]] = load {{.*}} @__safestack_unsafe_stack_ptr
  ; CHECK: %[[B:.*]] = ptrtoint ptr %[[A]] to i64
  ; CHECK: and i64 %[[B]], -64
  ; CHECK: ret i32
  %0 = load i32, ptr %zzz, align 64
  %arrayidx2 = getelementptr inbounds %struct.S, ptr %zzz, i64 0, i32 0, i64 %idx
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}


; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
;
; Check whether SafeStack recognizes 'unsafealloc' metadata.

define void @test() safestack {
  ; CHECK-LABEL: @test(
  ; CHECK: %[[SP:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
  ; CHECK: %[[STATICTOP:.*]] = getelementptr i8, ptr %[[SP]], i32 -16
  ; CHECK: store ptr %[[STATICTOP]], ptr @__safestack_unsafe_stack_ptr
  ; CHECK: %safe = alloca i32, align 4
  ; CHECK: store ptr %[[SP]], ptr @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %safe = alloca i32, align 4
  %unsafe = alloca i32, align 4, !unsafealloc !{}
  ret void
}


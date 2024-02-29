; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Addr-of a variable passed into an invoke instruction.
;  safestack attribute
; Requires protector and stack restore after landing pad.
define i32 @foo() uwtable safestack personality ptr @__gxx_personality_v0 {
entry:
  ; CHECK: %[[SP:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
  ; CHECK: %[[STATICTOP:.*]] = getelementptr i8, ptr %[[SP]], i32 -16
  %a = alloca i32, align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  store i32 0, ptr %a, align 4
  invoke void @_Z3exceptPi(ptr %a)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  ; CHECK: landingpad
  ; CHECK-NEXT: catch
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ; CHECK-NEXT: store ptr %[[STATICTOP]], ptr @__safestack_unsafe_stack_ptr
  ret i32 0
}

declare void @_Z3exceptPi(ptr)
declare i32 @__gxx_personality_v0(...)

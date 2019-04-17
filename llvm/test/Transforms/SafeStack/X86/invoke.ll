; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Addr-of a variable passed into an invoke instruction.
;  safestack attribute
; Requires protector and stack restore after landing pad.
define i32 @foo() uwtable safestack personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  ; CHECK: %[[SP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
  ; CHECK: %[[STATICTOP:.*]] = getelementptr i8, i8* %[[SP]], i32 -16
  %a = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i32 0, i32* %a, align 4
  invoke void @_Z3exceptPi(i32* %a)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  ; CHECK: landingpad
  ; CHECK-NEXT: catch
  %0 = landingpad { i8*, i32 }
          catch i8* null
  ; CHECK-NEXT: store i8* %[[STATICTOP]], i8** @__safestack_unsafe_stack_ptr
  ret i32 0
}

declare void @_Z3exceptPi(i32*)
declare i32 @__gxx_personality_v0(...)

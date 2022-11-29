; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

%struct.pair = type { i32, i32 }

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Addr-of a struct element passed into an invoke instruction.
;   (GEP followed by an invoke)
;  safestack attribute
; Requires protector.
define i32 @foo() uwtable safestack personality ptr @__gxx_personality_v0 {
entry:
  ; CHECK: __safestack_unsafe_stack_ptr
  %c = alloca %struct.pair, align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  store i32 0, ptr %c, align 4
  invoke void @_Z3exceptPi(ptr %c)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ret i32 0
}

declare void @_Z3exceptPi(ptr)
declare i32 @__gxx_personality_v0(...)

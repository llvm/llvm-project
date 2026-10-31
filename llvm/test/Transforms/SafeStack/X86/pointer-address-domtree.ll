; RUN: opt -mtriple=x86_64-pc-linux-gnu -safestack-use-pointer-address \
; RUN:   -passes='require<libcall-lowering-info>,function(require<domtree>,safe-stack,verify<domtree>)' \
; RUN:   -disable-output < %s
; RUN: opt -mtriple=x86_64-pc-linux-gnu -safestack-use-pointer-address \
; RUN:   -domtree -safe-stack -loops -verify-dom-info \
; RUN:   -disable-output < %s

@unsafe_stack_pointer_a = thread_local global ptr null
@unsafe_stack_pointer_b = thread_local global ptr null

declare i32 @get_runtime_mode()
declare void @escape(ptr)

define ptr @__safestack_pointer_address() alwaysinline {
entry:
  %first = call i32 @get_runtime_mode()
  %use_a = icmp eq i32 %first, 0
  br i1 %use_a, label %a, label %b

a:
  ret ptr @unsafe_stack_pointer_a

b:
  ret ptr @unsafe_stack_pointer_b
}

define i32 @caller() safestack {
entry:
  %unsafe = alloca i32, align 4
  call void @escape(ptr %unsafe)
  %second = call i32 @get_runtime_mode()
  ret i32 %second
}

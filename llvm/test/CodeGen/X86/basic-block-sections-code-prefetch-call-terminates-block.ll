;; Check prefetch directives properly handle a block terminating with a call.
;;
;; Specify the bb sections profile:
; RUN: echo 'v1' > %t
; RUN: echo 'f foo' >> %t
; RUN: echo 't 0,1' >> %t
;;
; RUN: llc < %s -mtriple=x86_64-pc-linux -asm-verbose=false -function-sections -basic-block-sections=%t -O1 | FileCheck %s

define i32 @foo() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @explode()
          to label %continue unwind label %cleanup
; CHECK:      foo:
; CHECK:        callq explode@PLT
; CHECK-NEXT:   .globl __llvm_prefetch_target_foo_0_1
; CHECK-NEXT: __llvm_prefetch_target_foo_0_1:

continue:
  ret i32 0

cleanup:
  %res = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %res
}

declare void @__gxx_personality_v0()
declare void @explode()

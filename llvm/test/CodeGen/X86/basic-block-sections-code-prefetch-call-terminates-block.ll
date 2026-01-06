;; Check prefetch directives properly handles prefetch targets and instructions after a call which terminates a block.
;;
;; Specify the bb sections profile:
; RUN: echo 'v1' > %t
; RUN: echo 'f fred' >> %t
; RUN: echo 't 0,1' >> %t
; RUN: echo 'i 0,1 other,30,30' >> %t
;;
; RUN: llc < %s -mtriple=x86_64-pc-linux -asm-verbose=false -function-sections -basic-block-sections=%t -O1 | FileCheck %s

define i32 @fred() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @explode()
          to label %continue unwind label %cleanup
; CHECK:      fred:
; CHECK:        callq explode@PLT
; CHECK-NEXT:   .globl __llvm_prefetch_target_fred_0_1
; CHECK-NEXT: __llvm_prefetch_target_fred_0_1:
; CHECK-NEXT:   prefetchit1 __llvm_prefetch_target_other_30_30(%rip)

continue:
  ret i32 0

cleanup:
  %res = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } %res
}

declare void @__gxx_personality_v0()
declare void @explode()

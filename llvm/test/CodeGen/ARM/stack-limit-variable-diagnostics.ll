; Because the -fstack-limit-variable stack check guards against stack overflow,
; a request the backend cannot honour is reported rather than silently dropped:
; an unsupported target, a malformed attribute or a function whose entry state
; the check sequence would corrupt is a hard error.

; RUN: rm -rf %t && split-file %s %t
; RUN: not --crash llc -mtriple=armv7-none-eabi %t/a32.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=A32
; RUN: not --crash llc -mtriple=thumbv6m-none-eabi -mattr=+execute-only %t/xo.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=XO
; RUN: not --crash llc -mtriple=thumbv7m-none-eabi %t/empty.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EMPTY
; RUN: not --crash llc -mtriple=thumbv7m-none-eabi %t/badtrap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=BADTRAP

; Both the Thumb2 and Thumb1 sequences use r12 (IP) as a scratch register, so
; a function where r12 is live on entry (it carries the 'nest' parameter) is
; rejected on both.
; RUN: not --crash llc -mtriple=thumbv7m-none-eabi %t/nest.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEST
; RUN: not --crash llc -mtriple=thumbv6m-none-eabi %t/nest.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEST

; A32: LLVM ERROR: {{.*}}-fstack-limit-variable is not supported for ARM (A32) code. Only Thumb (Cortex-M) targets are supported
; XO: LLVM ERROR: {{.*}}-fstack-limit-variable is not supported for execute-only ARMv6-M (Thumb1) code
; EMPTY: LLVM ERROR: {{.*}}'stack-limit-variable' attribute must name a non-empty global variable
; BADTRAP: LLVM ERROR: {{.*}}'stack-limit-trap-number' value '999' is not a valid supervisor-call number. Expected an integer in the range 0-255
; NEST: LLVM ERROR: {{.*}}-fstack-limit-variable uses r12 (IP) as a scratch register, but r12 cannot be proven free on entry to function 'f'. Functions taking a 'nest' parameter, which is passed in r12, are not supported

;--- a32.ll
define void @f() #0 { ret void }
attributes #0 = { "stack-limit-variable"="__stack_boundary" }

;--- xo.ll
define void @f() #0 { ret void }
attributes #0 = { "stack-limit-variable"="__stack_boundary" }

;--- empty.ll
define void @f() #0 { ret void }
attributes #0 = { "stack-limit-variable"="" }

;--- badtrap.ll
define void @f() #0 { ret void }
attributes #0 = { "stack-limit-variable"="__stack_boundary" "stack-limit-trap-number"="999" }

;--- nest.ll
declare void @sink(ptr)
define void @f(ptr nest %chain) #0 {
entry:
  %buf = alloca [64 x i8], align 1
  call void @sink(ptr %buf)
  call void @sink(ptr %chain)
  ret void
}
attributes #0 = { "stack-limit-variable"="__stack_boundary" }

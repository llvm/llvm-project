; Because the -fstack-limit-variable stack check guards against stack overflow,
; a request the backend cannot honour is reported rather than silently dropped:
; an unsupported target or a malformed attribute is a hard error.

; RUN: rm -rf %t && split-file %s %t
; RUN: not --crash llc -mtriple=armv7-none-eabi %t/a32.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=A32
; RUN: not --crash llc -mtriple=thumbv6m-none-eabi -mattr=+execute-only %t/xo.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=XO
; RUN: not --crash llc -mtriple=thumbv7m-none-eabi %t/empty.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EMPTY
; RUN: not --crash llc -mtriple=thumbv7m-none-eabi %t/badtrap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=BADTRAP

; A32: LLVM ERROR: {{.*}}-fstack-limit-variable is not supported for ARM (A32) code. Only Thumb (Cortex-M) targets are supported
; XO: LLVM ERROR: {{.*}}-fstack-limit-variable is not supported for execute-only ARMv6-M (Thumb1) code
; EMPTY: LLVM ERROR: {{.*}}'stack-limit-variable' attribute must name a non-empty global variable
; BADTRAP: LLVM ERROR: {{.*}}'stack-limit-trap-number' value '999' is not a valid supervisor-call number. Expected an integer in the range 0-255

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

; FIXME: this test fails with a stage2 build, in which case it seems to
; generate extra function prologue instructions that interfere with matching
; the STR of X19. Disable this for now.
;
; UNSUPPORTED: *

REQUIRES: aarch64-registered-target, asserts

RUN: llvm-exegesis -mcpu=neoverse-v2 --use-dummy-perf-counters --mode=latency --debug-only=print-gen-assembly --opcode-name=ADDVv4i16v -repetition-mode=loop 2>&1 | FileCheck %s

CHECK:       str     x19, [sp, #-16]!
CHECK-NEXT:  movi    d[[REG:[0-9]+]], #0000000000000000
CHECK-NEXT:  mov     x19, #10000
CHECK-NEXT:  nop
CHECK-NEXT:  nop
CHECK-NEXT:  nop
CHECK-NEXT:  nop
CHECK-NEXT:  nop
CHECK-NEXT:  addv    h[[REG]], v[[REG]].4h
CHECK-NEXT:  subs    x19, x19, #1
CHECK-NEXT:  b.ne    #-8
CHECK-NEXT:  ldr     x19, [sp], #16
CHECK-NEXT:  ret

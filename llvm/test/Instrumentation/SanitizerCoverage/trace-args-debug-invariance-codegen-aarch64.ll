; Debug information must not change the machine code of trace-args/trace-ret
; (AArch64 AAPCS64). The input is what clang -O2 -g emits for the argument shapes of
; Inputs/trace-args-cases-aarch64.ll; stripping its debug information first stands in
; for the same source built without -g. Code generation runs as clang runs it
; at -O2 -g, with call-site information.
;
; REQUIRES: aarch64-registered-target
;
; RUN: opt -passes='module(sancov-module)' -sanitizer-coverage-trace-args -sanitizer-coverage-trace-ret \
; RUN:   %S/Inputs/trace-args-cases-aarch64.ll -o %t.g.bc
; RUN: opt -strip-debug %S/Inputs/trace-args-cases-aarch64.ll \
; RUN:   | opt -passes='module(sancov-module)' -sanitizer-coverage-trace-args -sanitizer-coverage-trace-ret -o %t.nog.bc
; RUN: llc -O2 -filetype=obj -emit-call-site-info %t.g.bc -o %t.g.o
; RUN: llc -O2 -filetype=obj -emit-call-site-info %t.nog.bc -o %t.nog.o
;
; The debug build has debug information, the other has none.
; RUN: llvm-readobj --sections %t.g.o | FileCheck %s --check-prefix=G
; RUN: llvm-readobj --sections %t.nog.o | FileCheck %s --check-prefix=NOG
; G: Name: .debug_info
; NOG-NOT: Name: .debug_
;
; .text is byte-identical.
; RUN: llvm-objcopy --dump-section=.text=%t.g.text %t.g.o
; RUN: llvm-objcopy --dump-section=.text=%t.nog.text %t.nog.o
; RUN: cmp %t.g.text %t.nog.text
;
; And it does trace: the same calls in both, one per IR argument or
; first-class aggregate element, and one per return.
; RUN: llvm-objdump -d -r %t.g.o | FileCheck %s --check-prefix=ARGS
; RUN: llvm-objdump -d -r %t.nog.o | FileCheck %s --check-prefix=ARGS
; RUN: llvm-objdump -d -r %t.g.o | FileCheck %s --check-prefix=RET
; RUN: llvm-objdump -d -r %t.nog.o | FileCheck %s --check-prefix=RET
; ARGS-COUNT-37: R_{{[A-Z0-9_]+}} __sanitizer_cov_trace_args
; ARGS-NOT: __sanitizer_cov_trace_args
; RET-COUNT-16: R_{{[A-Z0-9_]+}} __sanitizer_cov_trace_ret
; RET-NOT: __sanitizer_cov_trace_ret

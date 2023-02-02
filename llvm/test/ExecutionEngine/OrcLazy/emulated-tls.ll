; LoongArch does not support emulated tls.
; UNSUPPORTED: target=loongarch{{.*}}

; RUN: not lli -no-process-syms -emulated-tls -jit-kind=orc-lazy %s 2>&1 \
; RUN:   | FileCheck %s
;
; Test that emulated-tls does not generate any unexpected errors.
;
; Unfortunately we cannot test successful execution of JIT'd code with
; emulated-tls as this would require the JIT itself, in this case lli, to be
; built with emulated-tls, which is not a common configuration. Instead we test
; that the only error produced by the JIT for a thread-local with emulated-tls
; enabled is a missing symbol error for __emutls_get_address. An unresolved
; reference to this symbol (and only this symbol) implies (1) that the emulated
; tls lowering was applied, and (2) that thread locals defined in the JIT'd code
; were otherwise handled correctly.

; CHECK: JIT session error: Symbols not found: [ {{[^,]*}}__emutls_get_address ]

@x = thread_local global i32 42, align 4

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %0 = load i32, ptr @x, align 4
  ret i32 %0
}

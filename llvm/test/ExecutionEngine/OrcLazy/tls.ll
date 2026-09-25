; LoongArch does not support emulated tls.
; UNSUPPORTED: target=loongarch{{.*}}

; RUN: not lli -no-process-syms -lljit-platform=Inactive -emulated-tls \
; RUN:   -jit-kind=orc-lazy %s 2>&1 | FileCheck %s --check-prefix=EMULATED
; RUN: not lli -no-process-syms -lljit-platform=Inactive -emulated-tls=false \
; RUN:   -jit-kind=orc-lazy %s 2>&1 | FileCheck %s --check-prefix=NATIVE
; RUN: %if system-linux %{ not lli -no-process-syms -lljit-platform=Inactive \
; RUN:   -emulated-tls=false -jit-kind=orc-lazy -jit-linker=rtdyld \
; RUN:   -code-model=small %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=RTDYLD %}
;
; Test that emulated and native TLS lowering produce the expected errors.
; Use the small code model for RuntimeDyld: AArch64's large code model does not
; support general-dynamic TLS lowering.
;
; TODO: Replace this test with positive tests of successful TLS execution using
; the new ORC runtime once it supports them.
;
; Unfortunately we cannot test successful execution of JIT'd code with
; emulated-tls as this would require the JIT itself, in this case lli, to be
; built with emulated-tls, which is not a common configuration. Instead we test
; that the only error produced by the JIT for a thread-local with emulated-tls
; enabled is a missing symbol error for __emutls_get_address. An unresolved
; reference to this symbol (and only this symbol) implies (1) that the emulated
; tls lowering was applied, and (2) that thread locals defined in the JIT'd code
; were otherwise handled correctly.

; EMULATED: JIT session error: Symbols not found: [ {{[^,]*}}__emutls_get_address ]
; NATIVE-NOT: __emutls_get_address
; NATIVE: JIT session error:
; NATIVE-NOT: __emutls_get_address
; RTDYLD: JIT session error: Unable to allocate TLS section memory

@x = thread_local global i32 42, align 4

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %0 = load i32, ptr @x, align 4
  ret i32 %0
}

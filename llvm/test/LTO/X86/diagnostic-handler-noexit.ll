; LTO default diagnostic handler should be non-exiting.
; This test verifies that after addModule() encounters an error, the diagnostic
; handler does not call exit(1) and instead returns to the caller of addModule.

; RUN: llvm-as <%s >%t1
; RUN: llvm-as <%s >%t2
; RUN: not llvm-lto -o /dev/null %t1 %t2 2>&1 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK: Linking globals named 'goodboy': symbol multiply defined!
; CHECK: llvm-lto{{.*}}: error adding file
@goodboy = global i32 3203383023, align 4    ; 0xbeefbeef

; RUN: llc -O2 --march=bpfel %s -o - | FileCheck %s

; Generated from the following C code:
;
;   extern int foo(void) __attribute__((bpf_fastcall));
;
;   int bar(int a, int b, int c, int d, int e) {
;     foo();
;     return e;
;   }
;
; Using the following command:
;
;   clang --target=bpf -emit-llvm -O2 -S -o - t.c
;
; (unnecessary attrs removed maually)

; Check that function marked with bpf_fastcall does not clobber W1-W5.

define dso_local i32 @bar(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
  %call = tail call i32 @foo() #0
  ret i32 %e
}

; CHECK:      # %bb.0:
; CHECK-NEXT:   *(u64 *)(r10 - 8) = r5
; CHECK-NEXT:   call foo
; CHECK-NEXT:   r5 = *(u64 *)(r10 - 8)
; CHECK-NEXT:   w0 = w5
; CHECK-NEXT:   exit

declare dso_local i32 @foo() #0

attributes #0 = { "bpf_fastcall" }

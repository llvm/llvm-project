; RUN: llc -O2 --mtriple=bpfel %s -o - | FileCheck %s

; Generated from the following C code:
;
;   extern int foo(void) __attribute__((bpf_fastcall));
;
;   int bar(int a) {
;     foo();
;     return a;
;   }
;
; Using the following command:
;
;   clang --target=bpf -emit-llvm -O2 -S -o - t.c
;
; (unnecessary attrs removed maually)

; Check that function marked with bpf_fastcall does not clobber W1-W5.

define dso_local i32 @bar(i32 %a) {
entry:
  %call = tail call i32 @foo() #0
  ret i32 %a
}

; CHECK:      # %bb.0:
; CHECK-NEXT:   *(u64 *)(r10 - 8) = r1
; CHECK-NEXT:   call foo
; CHECK-NEXT:   r1 = *(u64 *)(r10 - 8)
; CHECK-NEXT:   w0 = w1
; CHECK-NEXT:   exit

declare dso_local i32 @foo() #0

attributes #0 = { "bpf_fastcall" }

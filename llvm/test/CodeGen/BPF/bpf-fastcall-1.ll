; RUN: llc -O2 --march=bpfel %s -o - | FileCheck %s

; Generated from the following C code:
;
;   #define __bpf_fastcall __attribute__((bpf_fastcall))
;
;   void bar(void) __bpf_fastcall;
;   void buz(long i, long j, long k);
;
;   void foo(long i, long j, long k) {
;     bar();
;     buz(i, j, k);
;   }
;
; Using the following command:
;
;   clang --target=bpf -emit-llvm -O2 -S -o - t.c
;
; (unnecessary attrs removed maually)

; Check that function marked with bpf_fastcall does not clobber R1-R5.

define dso_local void @foo(i64 noundef %i, i64 noundef %j, i64 noundef %k) {
entry:
  tail call void @bar() #1
  tail call void @buz(i64 noundef %i, i64 noundef %j, i64 noundef %k)
  ret void
}

; CHECK:      foo:
; CHECK:      # %bb.0:
; CHECK-NEXT:   *(u64 *)(r10 - 8) = r1
; CHECK-NEXT:   *(u64 *)(r10 - 16) = r2
; CHECK-NEXT:   *(u64 *)(r10 - 24) = r3
; CHECK-NEXT:   call bar
; CHECK-NEXT:   r3 = *(u64 *)(r10 - 24)
; CHECK-NEXT:   r2 = *(u64 *)(r10 - 16)
; CHECK-NEXT:   r1 = *(u64 *)(r10 - 8)
; CHECK-NEXT:   call buz
; CHECK-NEXT:   exit

declare dso_local void @bar() #0
declare dso_local void @buz(i64 noundef, i64 noundef, i64 noundef)

attributes #0 = { "bpf_fastcall" }
attributes #1 = { nounwind "bpf_fastcall" }

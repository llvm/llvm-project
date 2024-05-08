; RUN: llc -O2 --march=bpfel %s -o - | FileCheck %s

; Generated from the following C code:
;
; #define __bpf_fastcall __attribute__((bpf_fastcall))
;
; void quux(void *);
; void bar(long) __bpf_fastcall;
; void buz(long i, long j);
;
; void foo(long i, long j) {
;   long k;
;   bar(i);
;   bar(i);
;   buz(i, j);
;   quux(&k);
; }
;
; Using the following command:
;
;   clang --target=bpf -emit-llvm -O2 -S -o - t.c
;
; (unnecessary attrs removed maually)

; Check that function marked with bpf_fastcall does not clobber R1-R5.
; Check that spills/fills wrapping the call use and reuse lowest stack offsets.

define dso_local void @foo(i64 noundef %i, i64 noundef %j) {
entry:
  %k = alloca i64, align 8
  tail call void @bar(i64 noundef %i) #0
  tail call void @bar(i64 noundef %i) #0
  tail call void @buz(i64 noundef %i, i64 noundef %j)
  call void @quux(ptr noundef nonnull %k)
  ret void
}

; CHECK:      # %bb.0:
; CHECK-NEXT:   r3 = r1
; CHECK-NEXT:   *(u64 *)(r10 - 16) = r2
; CHECK-NEXT:   *(u64 *)(r10 - 24) = r3
; CHECK-NEXT:   call bar
; CHECK-NEXT:   r3 = *(u64 *)(r10 - 24)
; CHECK-NEXT:   r2 = *(u64 *)(r10 - 16)
; CHECK-NEXT:   r1 = r3
; CHECK-NEXT:   *(u64 *)(r10 - 16) = r2
; CHECK-NEXT:   *(u64 *)(r10 - 24) = r3
; CHECK-NEXT:   call bar
; CHECK-NEXT:   r3 = *(u64 *)(r10 - 24)
; CHECK-NEXT:   r2 = *(u64 *)(r10 - 16)
; CHECK-NEXT:   r1 = r3
; CHECK-NEXT:   call buz
; CHECK-NEXT:   r1 = r10
; CHECK-NEXT:   r1 += -8
; CHECK-NEXT:   call quux
; CHECK-NEXT:   exit

declare dso_local void @bar(i64 noundef) #0
declare dso_local void @buz(i64 noundef, i64 noundef)
declare dso_local void @quux(ptr noundef)

attributes #0 = { "bpf_fastcall" }

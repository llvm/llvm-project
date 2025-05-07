; RUN: llc -O2 --mtriple=bpfel -mcpu=v1 %s -o - | FileCheck %s

; Generated from the following C code:
;
;   #define __bpf_fastcall __attribute__((bpf_fastcall))
;
;   void bar(void) __bpf_fastcall;
;   void buz(long i, long j);
;
;   void foo(long i, long j, long k, long l) {
;     bar();
;     if (k > 42l)
;       buz(i, 1);
;     else
;       buz(1, j);
;   }
;
; Using the following command:
;
;   clang --target=bpf -emit-llvm -O2 -S -o - t.c
;
; (unnecessary attrs removed maually)

; Check that function marked with bpf_fastcall does not clobber R1-R5.
; Use R1 in one branch following call and R2 in another branch following call.

define dso_local void @foo(i64 noundef %i, i64 noundef %j, i64 noundef %k, i64 noundef %l) {
entry:
  tail call void @bar() #0
  %cmp = icmp sgt i64 %k, 42
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @buz(i64 noundef %i, i64 noundef 1)
  br label %if.end

if.else:
  tail call void @buz(i64 noundef 1, i64 noundef %j)
  br label %if.end

if.end:
  ret void
}

; CHECK:      foo:                                    # @foo
; CHECK:      # %bb.0:                                # %entry
; CHECK-NEXT:   *(u64 *)(r10 - 8) = r1
; CHECK-NEXT:   *(u64 *)(r10 - 16) = r2
; CHECK-NEXT:   *(u64 *)(r10 - 24) = r3
; CHECK-NEXT:   call bar
; CHECK-NEXT:   r3 = *(u64 *)(r10 - 24)
; CHECK-NEXT:   r2 = *(u64 *)(r10 - 16)
; CHECK-NEXT:   r1 = *(u64 *)(r10 - 8)
; CHECK-NEXT:   r4 = 43
; CHECK-NEXT:   if r4 s> r3 goto [[ELSE:.*]]
; CHECK-NEXT: # %bb.1:                                # %if.then
; CHECK-NEXT:   r2 = 1
; CHECK-NEXT:   goto [[END:.*]]
; CHECK-NEXT: [[ELSE]]:                               # %if.else
; CHECK-NEXT:   r1 = 1
; CHECK-NEXT: [[END]]:                                # %if.end
; CHECK-NEXT:   call buz
; CHECK-NEXT:   exit

declare dso_local void @bar() #0
declare dso_local void @buz(i64 noundef, i64 noundef)

attributes #0 = { "bpf_fastcall" }

; RUN: llc -mtriple=arm64-unknown-unknown -mcpu=cyclone -pre-RA-sched=list-hybrid < %s | FileCheck %s
; rdar://10232252
; Prevent LSR of doing poor choice that cannot be folded in addressing mode

; Remove the -pre-RA-sched=list-hybrid option after fixing:
; <rdar://problem/12702735> [ARM64][coalescer] need better register
; coalescing for simple unit tests.

; CHECK: testCase
; CHECK: %while.body{{$}}
; CHECK: ldr [[STREG:x[0-9]+]], [{{x[0-9]+}}], #8
; CHECK-NEXT: str [[STREG]], [{{x[0-9]+}}], #8
; CHECK: %while.end
define i32 @testCase() nounwind ssp {
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %len.06 = phi i64 [ 1288, %entry ], [ %sub, %while.body ]
  %pDst.05 = phi i64* [ inttoptr (i64 6442450944 to i64*), %entry ], [ %incdec.ptr1, %while.body ]
  %pSrc.04 = phi i64* [ inttoptr (i64 4294967296 to i64*), %entry ], [ %incdec.ptr, %while.body ]
  %incdec.ptr = getelementptr inbounds i64, i64* %pSrc.04, i64 1
  %tmp = load volatile i64, i64* %pSrc.04, align 8
  %incdec.ptr1 = getelementptr inbounds i64, i64* %pDst.05, i64 1
  store volatile i64 %tmp, i64* %pDst.05, align 8
  %sub = add i64 %len.06, -8
  %cmp = icmp sgt i64 %sub, -1
  br i1 %cmp, label %while.body, label %while.end

while.end:                                        ; preds = %while.body
  tail call void inttoptr (i64 6442450944 to void ()*)() nounwind
  ret i32 0
}

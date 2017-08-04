; RUN: opt %loadPolly -analyze -polly-detect \
; RUN:     -pass-remarks-missed="polly-detect" \
; RUN:     < %s 2>&1| FileCheck %s

; CHECK: remark: <unknown>:0:0: Loop cannot be handled because not all latches are part of loop region.

define void @foo(i8* %str0) {
if.end32:
  br label %while.cond

while.cond:
  %str.1 = phi i8* [%str0, %if.end32], [%incdec.ptr58364, %lor.end], [%incdec.ptr58364, %while.cond]
  %tmp5 = load i8, i8* %str.1, align 1
  %.off367 = add i8 %tmp5, -48
  %tmp6 = icmp ult i8 %.off367, 10
  %incdec.ptr58364 = getelementptr inbounds i8, i8* %str.1, i64 1
  br i1 %tmp6, label %while.cond, label %lor.end

lor.end:
  br i1 false, label %exit, label %while.cond

exit:
  ret void
}

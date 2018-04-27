; RUN: opt -S -licm < %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' -S %s | FileCheck %s

declare void @use_nothrow(i64 %a) nounwind
declare void @use(i64 %a)
declare void @maythrow()

define void @nothrow(i64 %x, i64 %y, i1* %cond) {
; CHECK-LABEL: nothrow
; CHECK-LABEL: entry
; CHECK: %div = udiv i64 %x, %y
; CHECK-LABEL: loop
; CHECK: call void @use_nothrow(i64 %div)
entry:
  br label %loop

loop:                                         ; preds = %entry, %for.inc
  %div = udiv i64 %x, %y
  br label %loop2

loop2:
  call void @use_nothrow(i64 %div)
  br label %loop
}

; The udiv is guarantee to execute if the loop is
define void @throw_header_after(i64 %x, i64 %y, i1* %cond) {
; CHECK-LABEL: throw_header_after
; CHECK: %div = udiv i64 %x, %y
; CHECK-LABEL: loop
; CHECK: call void @use(i64 %div)
entry:
  br label %loop

loop:                                         ; preds = %entry, %for.inc
  %div = udiv i64 %x, %y
  call void @use(i64 %div)
  br label %loop
}
define void @throw_header_after_rec(i64* %xp, i64* %yp, i1* %cond) {
; CHECK-LABEL: throw_header_after_rec
; CHECK: %x = load i64, i64* %xp
; CHECK: %y = load i64, i64* %yp
; CHECK: %div = udiv i64 %x, %y
; CHECK-LABEL: loop
; CHECK: call void @use(i64 %div)
entry:
  br label %loop

loop:                                         ; preds = %entry, %for.inc
  %x = load i64, i64* %xp
  %y = load i64, i64* %yp
  %div = udiv i64 %x, %y
  call void @use(i64 %div) readonly
  br label %loop
}



; Negative test
define void @throw_header_before(i64 %x, i64 %y, i1* %cond) {
; CHECK-LABEL: throw_header_before
; CHECK-LABEL: loop
; CHECK: %div = udiv i64 %x, %y
; CHECK: call void @use(i64 %div)
entry:
  br label %loop

loop:                                         ; preds = %entry, %for.inc
  call void @maythrow()
  %div = udiv i64 %x, %y
  call void @use(i64 %div)
  br label %loop
}

; The header is known no throw, but the loop is not.  We can
; still lift out of the header.
define void @nothrow_header(i64 %x, i64 %y, i1 %cond) {
; CHECK-LABEL: nothrow_header
; CHECK-LABEL: entry
; CHECK: %div = udiv i64 %x, %y
; CHECK-LABEL: loop
  ; CHECK: call void @use(i64 %div)
entry:
  br label %loop
loop:                                         ; preds = %entry, %for.inc
  %div = udiv i64 %x, %y
  br i1 %cond, label %loop-if, label %exit
loop-if:
  call void @use(i64 %div)
  br label %loop
exit:
  ret void
}
; Negative test - can't move out of throwing block
define void @nothrow_header_neg(i64 %x, i64 %y, i1 %cond) {
; CHECK-LABEL: nothrow_header_neg
; CHECK-LABEL: entry
; CHECK-LABEL: loop
; CHECK: %div = udiv i64 %x, %y
; CHECK: call void @use(i64 %div)
entry:
  br label %loop
loop:                                         ; preds = %entry, %for.inc
  br label %loop-if
loop-if:
  %div = udiv i64 %x, %y
  call void @use(i64 %div)
  br label %loop
}

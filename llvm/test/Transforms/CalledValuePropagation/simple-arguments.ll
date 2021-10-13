; RUN: opt -passes=called-value-propagation -S < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnueabi"


; This test checks that we propagate the functions through arguments and attach
; !callees metadata to the call. Such metadata can enable optimizations of this
; code sequence.
;
; For example, the code below a illustrates a contrived sort-like algorithm
; that accepts a pointer to a comparison function. Since the indirect call to
; the comparison function has only two targets, the call can be promoted to two
; direct calls using an if-then-else. The loop can then be unswitched and the
; called functions inlined. This essentially produces two loops, once
; specialized for each comparison.
;
; CHECK:  %tmp3 = call i1 %cmp(i64* %tmp1, i64* %tmp2), !callees ![[MD:[0-9]+]]
; CHECK: ![[MD]] = !{i1 (i64*, i64*)* @ugt, i1 (i64*, i64*)* @ule}
;
define void @test_argument(i64* %x, i64 %n, i1 %flag) {
entry:
  %tmp0 = sub i64 %n, 1
  br i1 %flag, label %then, label %else

then:
  call void @arrange_data(i64* %x, i64 %tmp0, i1 (i64*, i64*)* @ugt)
  br label %merge

else:
  call void @arrange_data(i64* %x, i64 %tmp0, i1 (i64*, i64*)* @ule)
  br label %merge

merge:
  ret void
}

define internal void @arrange_data(i64* %x, i64 %n, i1 (i64*, i64*)* %cmp) {
entry:
  %tmp0 = icmp eq i64 %n, 1
  br i1 %tmp0, label %merge, label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %cmp.false ]
  %i.next = add nuw nsw i64 %i, 1
  %tmp1 = getelementptr inbounds i64, i64* %x, i64 %i
  %tmp2 = getelementptr inbounds i64, i64* %x, i64 %i.next
  %tmp3 = call i1 %cmp(i64* %tmp1, i64* %tmp2)
  br i1 %tmp3, label %cmp.true, label %cmp.false

cmp.true:
  call void @swap(i64* %tmp1, i64* %tmp2)
  br label %cmp.false

cmp.false:
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp4 = sub i64 %n, 1
  call void @arrange_data(i64* %x, i64 %tmp4, i1 (i64*, i64*)* %cmp)
  br label %merge

merge:
  ret void
}

define internal i1 @ugt(i64* %a, i64* %b) {
entry:
  %tmp0 = load i64, i64* %a
  %tmp1 = load i64, i64* %b
  %tmp2 = icmp ugt i64 %tmp0, %tmp1
  ret i1 %tmp2
}

define internal i1 @ule(i64* %a, i64* %b) {
entry:
  %tmp0 = load i64, i64* %a
  %tmp1 = load i64, i64* %b
  %tmp2 = icmp ule i64 %tmp0, %tmp1
  ret i1 %tmp2
}

declare void @swap(i64*, i64*)

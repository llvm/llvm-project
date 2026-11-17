; REQUIRES: asserts
; RUN: opt %s -passes='print<delinearization>' -debug-only=delinearize -disable-output 2>&1 | FileCheck  %s

; CHECK-LABEL: Printing analysis 'Delinearization' for function 'symbolic_trip_count':
; CHECK: Range check via trip count: %cols <=s %cols
; CHECK: Delinearization validation: Failed

define void @symbolic_trip_count(ptr %a, i64 %cols) {
entry:
  %positive = icmp sgt i64 %cols, 0
  call void @llvm.assume(i1 %positive)
  br label %row.header

row.header:
  %row = phi i64 [ 0, %entry ], [ %row.next, %row.latch ]
  %row.off = mul nuw nsw i64 %row, %cols
  br label %col.header

col.header:
  %col = phi i64 [ 0, %row.header ], [ %col.next, %col.header ]
  %off = add nuw nsw i64 %row.off, %col
  %ptr = getelementptr nusw nuw i8, ptr %a, i64 %off
  store i8 0, ptr %ptr, align 1
  %col.next = add nuw nsw i64 %col, 1
  %col.done = icmp eq i64 %col.next, %cols
  br i1 %col.done, label %row.latch, label %col.header

row.latch:
  %row.next = add nuw nsw i64 %row, 1
  %row.done = icmp eq i64 %row.next, 2
  br i1 %row.done, label %exit, label %row.header

exit:
  ret void
}

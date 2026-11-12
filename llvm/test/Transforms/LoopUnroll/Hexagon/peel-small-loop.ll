; RUN: opt -passes=loop-unroll -mtriple=hexagon -S < %s | FileCheck %s
; Check that the loop is peeled twice for Hexagon.
; CHECK: while.body.peel
; CHECK: while.body.peel2

%struct.STREAM = type { %union.anon, i32, i32 }
%union.anon = type { ptr }

define void @function(ptr nocapture readonly %b) local_unnamed_addr {
entry:
  %bitPtr3 = getelementptr inbounds %struct.STREAM, ptr %b, i32 0, i32 2
  %0 = load i32, ptr %bitPtr3, align 4
  %cmp11 = icmp ult i32 %0, 32
  br i1 %cmp11, label %while.body.preheader, label %do.end

while.body.preheader:
  %value2 = getelementptr inbounds %struct.STREAM, ptr %b, i32 0, i32 1
  %1 = load i32, ptr %value2, align 4
  %2 = load ptr, ptr %b, align 4
  br label %while.body

while.body:
  %bitPtr.014 = phi i32 [ %add, %while.body ], [ %0, %while.body.preheader ]
  %value.013 = phi i32 [ %shl, %while.body ], [ %1, %while.body.preheader ]
  %ptr.012 = phi ptr [ %incdec.ptr, %while.body ], [ %2, %while.body.preheader ]
  %add = add nuw i32 %bitPtr.014, 8
  %shr = lshr i32 %value.013, 24
  %incdec.ptr = getelementptr inbounds i32, ptr %ptr.012, i32 1
  store i32 %shr, ptr %ptr.012, align 4
  %shl = shl i32 %value.013, 8
  %cmp = icmp ult i32 %add, 17
  br i1 %cmp, label %while.body, label %do.end

do.end:
  ret void
}

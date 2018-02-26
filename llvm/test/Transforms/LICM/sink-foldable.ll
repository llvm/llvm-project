; REQUIRES: aarch64-registered-target

; RUN: opt < %s  -licm -S   | FileCheck %s

target triple = "aarch64--linux-gnueabi"

; CHECK-LABEL:@test1
; CHECK-LABEL:loopexit1:
; CHECK: %[[PHI:.+]] = phi i8** [ %arrayidx0, %if.end ]
; CHECK: getelementptr inbounds i8*, i8** %[[PHI]], i64 1
define i8** @test1(i32 %j, i8** readonly %P, i8* readnone %Q) {
entry:
  %cmp0 = icmp slt i32 0, %j
  br i1 %cmp0, label %for.body.lr.ph, label %return

for.body.lr.ph:
  br label %for.body

for.body:
  %P.addr = phi i8** [ %P, %for.body.lr.ph ], [ %arrayidx0, %if.end  ]
  %i0 = phi i32 [ 0, %for.body.lr.ph ], [ %i.add, %if.end]

  %i0.ext = sext i32 %i0 to i64
  %arrayidx0 = getelementptr inbounds i8*, i8** %P.addr, i64 %i0.ext
  %l0 = load i8*, i8** %arrayidx0, align 8
  %cmp1 = icmp ugt i8* %l0, %Q
  br i1 %cmp1, label %loopexit0, label %if.end

if.end:                                           ; preds = %for.body
  %arrayidx1 = getelementptr inbounds i8*, i8** %arrayidx0, i64 1
  %l1 = load i8*, i8** %arrayidx1, align 8
  %cmp4 = icmp ugt i8* %l1, %Q
  %i.add = add nsw i32 %i0, 2
  br i1 %cmp4, label %loopexit1, label %for.body

loopexit0:
  %p1 = phi i8** [%arrayidx0, %for.body]
  br label %return

loopexit1:
  %p2 = phi i8** [%arrayidx1, %if.end]
  br label  %return

return:
  %retval.0 = phi i8** [ %p1, %loopexit0 ], [%p2, %loopexit1], [ null, %entry ]
  ret i8** %retval.0
}

; CHECK-LABEL: @test2
; CHECK-LABEL: loopexit2:
; CHECK: %[[PHI:.*]] = phi i8** [ %add.ptr, %if.end ]
; CHECK: getelementptr inbounds i8*, i8** %[[PHI]]
define i8** @test2(i32 %j, i8** readonly %P, i8* readnone %Q) {

entry:
  br label %for.body

for.cond:
  %i.addr.0 = phi i32 [ %add, %if.end ]
  %P.addr.0 = phi i8** [ %add.ptr, %if.end ]
  %cmp = icmp slt i32 %i.addr.0, %j
  br i1 %cmp, label %for.body, label %loopexit0

for.body:
  %P.addr = phi i8** [ %P, %entry ], [ %P.addr.0, %for.cond ]
  %i.addr = phi i32 [ 0, %entry ], [ %i.addr.0, %for.cond ]

  %idx.ext = sext i32 %i.addr to i64
  %add.ptr = getelementptr inbounds i8*, i8** %P.addr, i64 %idx.ext
  %l0 = load i8*, i8** %add.ptr, align 8

  %cmp1 = icmp ugt i8* %l0, %Q
  br i1 %cmp1, label %loopexit1, label %if.end

if.end:
  %add.i = add i32 %i.addr, 1
  %idx2.ext = sext i32 %add.i to i64
  %arrayidx2 = getelementptr inbounds i8*, i8** %add.ptr, i64 %idx2.ext
  %l1 = load i8*, i8** %arrayidx2, align 8
  %cmp2 = icmp ugt i8* %l1, %Q
  %add = add nsw i32 %add.i, 1
  br i1 %cmp2, label %loopexit2, label %for.cond

loopexit0:
  %p0 = phi i8** [ null, %for.cond ]
  br label %return

loopexit1:
  %p1 = phi i8** [ %add.ptr, %for.body ]
  br label %return

loopexit2:
  %p2 = phi i8** [ %arrayidx2, %if.end ]
  br label %return

return:
  %retval.0 = phi i8** [ %p1, %loopexit1 ], [ %p2, %loopexit2 ], [ %p0, %loopexit0 ]
  ret i8** %retval.0
}


; CHECK-LABEL: @test3
; CHECK-LABEL: loopexit1:
; CHECK: %[[ADD:.*]]  = phi i64 [ %add, %if.end ]
; CHECK: %[[ADDR:.*]] = phi i8** [ %P.addr, %if.end ]
; CHECK: %[[TRUNC:.*]] = trunc i64 %[[ADD]] to i32
; CHECK: getelementptr inbounds i8*, i8** %[[ADDR]], i32 %[[TRUNC]]
; CHECK: call void @dummy(i32 %[[TRUNC]])
define i8** @test3(i64 %j, i8** readonly %P, i8* readnone %Q) {
entry:
  %cmp0 = icmp slt i64 0, %j
  br i1 %cmp0, label %for.body.lr.ph, label %return

for.body.lr.ph:
  br label %for.body

for.body:
  %P.addr = phi i8** [ %P, %for.body.lr.ph ], [ %arrayidx0, %if.end  ]
  %i0 = phi i32 [ 0, %for.body.lr.ph ], [ %i.add, %if.end]

  %i0.ext = sext i32 %i0 to i64
  %arrayidx0 = getelementptr inbounds i8*, i8** %P.addr, i64 %i0.ext
  %l0 = load i8*, i8** %arrayidx0, align 8
  %cmp1 = icmp ugt i8* %l0, %Q
  br i1 %cmp1, label %loopexit0, label %if.end

if.end:                                           ; preds = %for.body
  %add = add i64 %i0.ext, 1
  %trunc = trunc i64 %add to i32
  %arrayidx1 = getelementptr inbounds i8*, i8** %P.addr, i32 %trunc
  %l1 = load i8*, i8** %arrayidx1, align 8
  %cmp4 = icmp ugt i8* %l1, %Q
  %i.add = add nsw i32 %i0, 2
  br i1 %cmp4, label %loopexit1, label %for.body

loopexit0:
  %p1 = phi i8** [%arrayidx0, %for.body]
  br label %return

loopexit1:
  %p2 = phi i8** [%arrayidx1, %if.end]
  call void @dummy(i32 %trunc)
  br label  %return

return:
  %retval.0 = phi i8** [ %p1, %loopexit0 ], [%p2, %loopexit1], [ null, %entry ]
  ret i8** %retval.0
}

declare void @dummy(i32)

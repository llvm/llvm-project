; RUN: llc -O3 -aarch64-enable-gep-opt=true  -print-after=codegenprepare -mcpu=cortex-a53 < %s >%t 2>&1 && FileCheck <%t %s
; REQUIRES: asserts
target triple = "aarch64--linux-android"

%typeD = type { i32, i32, [256 x i32], [257 x i32] }

; Function Attrs: noreturn nounwind uwtable
define i32 @test1(%typeD* nocapture %s) {
entry:
; CHECK-LABEL: entry:
; CHECK:    %uglygep = getelementptr i8, i8* %0, i64 1032
; CHECK:    br label %do.body.i


  %tPos = getelementptr inbounds %typeD, %typeD* %s, i64 0, i32 0
  %k0 = getelementptr inbounds %typeD, %typeD* %s, i64 0, i32 1
  %.pre = load i32, i32* %tPos, align 4
  br label %do.body.i

do.body.i:
; CHECK-LABEL: do.body.i:
; CHECK:          %uglygep2 = getelementptr i8, i8* %uglygep, i64 %3
; CHECK-NEXT:     %4 = bitcast i8* %uglygep2 to i32*
; CHECK-NOT:      %uglygep2 = getelementptr i8, i8* %uglygep, i64 1032


  %0 = phi i32 [ 256, %entry ], [ %.be, %do.body.i.backedge ]
  %1 = phi i32 [ 0, %entry ], [ %.be6, %do.body.i.backedge ]
  %add.i = add nsw i32 %1, %0
  %shr.i = ashr i32 %add.i, 1
  %idxprom.i = sext i32 %shr.i to i64
  %arrayidx.i = getelementptr inbounds %typeD, %typeD* %s, i64 0, i32 3, i64 %idxprom.i
  %2 = load i32, i32* %arrayidx.i, align 4
  %cmp.i = icmp sle i32 %2, %.pre
  %na.1.i = select i1 %cmp.i, i32 %0, i32 %shr.i
  %nb.1.i = select i1 %cmp.i, i32 %shr.i, i32 %1
  %sub.i = sub nsw i32 %na.1.i, %nb.1.i
  %cmp1.i = icmp eq i32 %sub.i, 1
  br i1 %cmp1.i, label %fooo.exit, label %do.body.i.backedge

do.body.i.backedge:
  %.be = phi i32 [ %na.1.i, %do.body.i ], [ 256, %fooo.exit ]
  %.be6 = phi i32 [ %nb.1.i, %do.body.i ], [ 0, %fooo.exit ]
  br label %do.body.i

fooo.exit:                              ; preds = %do.body.i
  store i32 %nb.1.i, i32* %k0, align 4
  br label %do.body.i.backedge
}


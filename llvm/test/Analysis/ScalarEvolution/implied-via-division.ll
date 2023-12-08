; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

define void @test_1(i32 %n) nounwind {
; Prove that (n > 1) ===> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @test_1
; CHECK:         Loop %header: backedge-taken count is (-1 + %n.div.2)<nsw>
entry:
  %cmp1 = icmp sgt i32 %n, 1
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sgt i32 %n.div.2, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_1neg(i32 %n) nounwind {
; Prove that (n > 0) =\=> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @test_1neg
; CHECK:         Loop %header: backedge-taken count is (-1 + (1 smax %n.div.2))<nsw>
entry:
  %cmp1 = icmp sgt i32 %n, 0
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sgt i32 %n.div.2, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_2(i32 %n) nounwind {
; Prove that (n >= 2) ===> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @test_2
; CHECK:         Loop %header: backedge-taken count is (-1 + %n.div.2)<nsw>
entry:
  %cmp1 = icmp sge i32 %n, 2
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sgt i32 %n.div.2, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_2neg(i32 %n) nounwind {
; Prove that (n >= 1) =\=> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @test_2neg
; CHECK:         Loop %header: backedge-taken count is (-1 + (1 smax %n.div.2))<nsw>
entry:
  %cmp1 = icmp sge i32 %n, 1
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sgt i32 %n.div.2, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_3(i32 %n) nounwind {
; Prove that (n > -2) ===> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @test_3
; CHECK:         Loop %header: backedge-taken count is (1 + %n.div.2)<nsw>
entry:
  %cmp1 = icmp sgt i32 %n, -2
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sge i32 %n.div.2, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_3neg(i32 %n) nounwind {
; Prove that (n > -3) =\=> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @test_3neg
; CHECK:         Loop %header: backedge-taken count is (0 smax (1 + %n.div.2)<nsw>)
entry:
  %cmp1 = icmp sgt i32 %n, -3
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sge i32 %n.div.2, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_4(i32 %n) nounwind {
; Prove that (n >= -1) ===> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @test_4
; CHECK:         Loop %header: backedge-taken count is (1 + %n.div.2)<nsw>
entry:
  %cmp1 = icmp sge i32 %n, -1
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sge i32 %n.div.2, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_4neg(i32 %n) nounwind {
; Prove that (n >= -2) =\=> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @test_4neg
; CHECK:         Loop %header: backedge-taken count is (0 smax (1 + %n.div.2)<nsw>)
entry:
  %cmp1 = icmp sge i32 %n, -2
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sge i32 %n.div.2, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_ext_01(i32 %n) nounwind {
; Prove that (n > 1) ===> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @test_ext_01
; CHECK:         Loop %header: backedge-taken count is (-1 + (sext i32 %n.div.2 to i64))<nsw>
entry:
  %cmp1 = icmp sgt i32 %n, 1
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sgt i64 %n.div.2.ext, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_ext_01neg(i32 %n) nounwind {
; Prove that (n > 0) =\=> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @test_ext_01neg
; CHECK:         Loop %header: backedge-taken count is (-1 + (1 smax (sext i32 %n.div.2 to i64)))<nsw>
entry:
  %cmp1 = icmp sgt i32 %n, 0
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sgt i64 %n.div.2.ext, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_ext_02(i32 %n) nounwind {
; Prove that (n >= 2) ===> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @test_ext_02
; CHECK:         Loop %header: backedge-taken count is (-1 + (sext i32 %n.div.2 to i64))<nsw>
entry:
  %cmp1 = icmp sge i32 %n, 2
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sgt i64 %n.div.2.ext, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_ext_02neg(i32 %n) nounwind {
; Prove that (n >= 1) =\=> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @test_ext_02neg
; CHECK:         Loop %header: backedge-taken count is (-1 + (1 smax (sext i32 %n.div.2 to i64)))<nsw>
entry:
  %cmp1 = icmp sge i32 %n, 1
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sgt i64 %n.div.2.ext, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_ext_03(i32 %n) nounwind {
; Prove that (n > -2) ===> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @test_ext_03
; CHECK:         Loop %header: backedge-taken count is (1 + (sext i32 %n.div.2 to i64))<nsw>
entry:
  %cmp1 = icmp sgt i32 %n, -2
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sge i64 %n.div.2.ext, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_ext_03neg(i32 %n) nounwind {
; Prove that (n > -3) =\=> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @test_ext_03neg
; CHECK:         Loop %header: backedge-taken count is (0 smax (1 + (sext i32 %n.div.2 to i64))<nsw>)
entry:
  %cmp1 = icmp sgt i32 %n, -3
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sge i64 %n.div.2.ext, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_ext_04(i32 %n) nounwind {
; Prove that (n >= -1) ===> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @test_ext_04
; CHECK:         Loop %header: backedge-taken count is (1 + (sext i32 %n.div.2 to i64))<nsw>
entry:
  %cmp1 = icmp sge i32 %n, -1
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sge i64 %n.div.2.ext, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test_ext_04neg(i32 %n) nounwind {
; Prove that (n >= -2) =\=> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @test_ext_04neg
; CHECK:         Loop %header: backedge-taken count is (0 smax (1 + (sext i32 %n.div.2 to i64))<nsw>)
entry:
  %cmp1 = icmp sge i32 %n, -2
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sge i64 %n.div.2.ext, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

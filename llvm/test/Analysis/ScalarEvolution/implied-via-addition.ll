; RUN: opt -indvars -S < %s | FileCheck %s

declare void @use(i1)

declare void @llvm.experimental.guard(i1, ...)

define void @test_01(i8 %t) {
; CHECK-LABEL: test_01
 entry:
  %st = sext i8 %t to i16
  %cmp1 = icmp slt i16 %st, 42
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %loop

 loop:
; CHECK-LABEL: loop
  %idx = phi i8 [ %t, %entry ], [ %idx.inc, %loop ]
  %idx.inc = add i8 %idx, 1
  %c = icmp slt i8 %idx, 42
; CHECK: call void @use(i1 true)
  call void @use(i1 %c)
  %be = icmp slt i8 %idx.inc, 42
  br i1 %be, label %loop, label %exit

 exit:
  ret void
}

define void @test_02(i8 %t) {
; CHECK-LABEL: test_02
 entry:
  %t.ptr = inttoptr i8 %t to i8*
  %p.42 = inttoptr i8 42 to i8*
  %cmp1 = icmp slt i8* %t.ptr, %p.42
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %loop

 loop:
; CHECK-LABEL: loop
  %idx = phi i8* [ %t.ptr, %entry ], [ %snext, %loop ]
  %snext = getelementptr inbounds i8, i8* %idx, i64 1
  %c = icmp slt i8* %idx, %p.42
; CHECK: call void @use(i1 true)
  call void @use(i1 %c)
  %be = icmp slt i8* %snext, %p.42
  br i1 %be, label %loop, label %exit

 exit:
  ret void
}

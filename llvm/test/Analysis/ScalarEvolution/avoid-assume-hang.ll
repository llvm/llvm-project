; RUN: opt %s -always-inline | opt -passes='print<scalar-evolution>'
; There was optimization bug in ScalarEvolution, that causes too long 
; compute time and stack overflow crash.

declare void @body(i32)
declare void @llvm.assume(i1)

define available_externally void @assume1(i64 %i.ext, i64 %a) alwaysinline {
  %cmp0 = icmp ne i64 %i.ext, %a
  call void @llvm.assume(i1 %cmp0)

  %a1 = add i64 %a, 1
  %cmp1 = icmp ne i64 %i.ext, %a1
  call void @llvm.assume(i1 %cmp1)

  %a2 = add i64 %a1, 1
  %cmp2 = icmp ne i64 %i.ext, %a2
  call void @llvm.assume(i1 %cmp2)

  %a3 = add i64 %a2, 1
  %cmp3 = icmp ne i64 %i.ext, %a3
  call void @llvm.assume(i1 %cmp3)

  %a4 = add i64 %a3, 1
  %cmp4 = icmp ne i64 %i.ext, %a4
  call void @llvm.assume(i1 %cmp4)

  ret void
}

define available_externally void @assume2(i64 %i.ext, i64 %a) alwaysinline {
  call void @assume1(i64 %i.ext, i64 %a)

  %a1 = add i64 %a, 5
  %cmp1 = icmp ne i64 %i.ext, %a1
  call void @assume1(i64 %i.ext, i64 %a1)

  %a2 = add i64 %a1, 5
  %cmp2 = icmp ne i64 %i.ext, %a2
  call void @assume1(i64 %i.ext, i64 %a2)

  %a3 = add i64 %a2, 5
  %cmp3 = icmp ne i64 %i.ext, %a3
  call void @assume1(i64 %i.ext, i64 %a3)

  %a4 = add i64 %a3, 5
  %cmp4 = icmp ne i64 %i.ext, %a4
  call void @assume1(i64 %i.ext, i64 %a4)

  ret void
}

define available_externally void @assume3(i64 %i.ext, i64 %a) alwaysinline {
  call void @assume2(i64 %i.ext, i64 %a)

  %a1 = add i64 %a, 25
  %cmp1 = icmp ne i64 %i.ext, %a1
  call void @assume2(i64 %i.ext, i64 %a1)

  %a2 = add i64 %a1, 25
  %cmp2 = icmp ne i64 %i.ext, %a2
  call void @assume2(i64 %i.ext, i64 %a2)

  %a3 = add i64 %a2, 25
  %cmp3 = icmp ne i64 %i.ext, %a3
  call void @assume2(i64 %i.ext, i64 %a3)

  %a4 = add i64 %a3, 25
  %cmp4 = icmp ne i64 %i.ext, %a4
  call void @assume2(i64 %i.ext, i64 %a4)

  ret void
}

define available_externally void @assume4(i64 %i.ext, i64 %a) alwaysinline {
  call void @assume3(i64 %i.ext, i64 %a)

  %a1 = add i64 %a, 125
  %cmp1 = icmp ne i64 %i.ext, %a1
  call void @assume3(i64 %i.ext, i64 %a1)

  %a2 = add i64 %a1, 125
  %cmp2 = icmp ne i64 %i.ext, %a2
  call void @assume3(i64 %i.ext, i64 %a2)

  %a3 = add i64 %a2, 125
  %cmp3 = icmp ne i64 %i.ext, %a3
  call void @assume3(i64 %i.ext, i64 %a3)

  %a4 = add i64 %a3, 125
  %cmp4 = icmp ne i64 %i.ext, %a4
  call void @assume3(i64 %i.ext, i64 %a4)

  ret void
}

define available_externally void @assume5(i64 %i.ext, i64 %a) alwaysinline {
  call void @assume4(i64 %i.ext, i64 %a)

  %a1 = add i64 %a, 625
  %cmp1 = icmp ne i64 %i.ext, %a1
  call void @assume4(i64 %i.ext, i64 %a1)

  %a2 = add i64 %a1, 625
  %cmp2 = icmp ne i64 %i.ext, %a2
  call void @assume4(i64 %i.ext, i64 %a2)

  %a3 = add i64 %a2, 625
  %cmp3 = icmp ne i64 %i.ext, %a3
  call void @assume4(i64 %i.ext, i64 %a3)

  %a4 = add i64 %a3, 625
  %cmp4 = icmp ne i64 %i.ext, %a4
  call void @assume4(i64 %i.ext, i64 %a4)

  ret void
}

define void @fn(i32 %init) {
entry:
  br label %loop

loop:
  %i = phi i32 [%init, %entry], [%next, %loop]
  call void @body(i32 %i)

  %i.ext = zext i32 %i to i64

  call void @assume5(i64 %i.ext, i64 500000000)

  %i.next = add i64 %i.ext, 1
  %next = trunc i64 %i.next to i32
  %done = icmp eq i32 %i, 500000000

  br i1 %done, label %exit, label %loop

exit:
  ret void
}

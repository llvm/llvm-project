; RUN: llc < %s -mtriple=xcore

define void @f( ) {
entry:

  switch i32 undef, label %default [
    i32 0, label %start
  ]

start:
  br label %end

default:
  %arg = fadd double undef, undef
  %res = call double @f2(i32 undef, double %arg, double undef)
  br label %end

end:
  %unused = phi double [ %res, %default ], [ undef, %start ]

  unreachable
}

declare double @f2(i32, double, double)


; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep load | count 1

; Instcombine should be able to do trivial CSE of loads.

declare void @use(double %n)
define void @bar(double* %p) {
  %t0 = getelementptr double* %p, i32 1
  %y = load double* %t0
  %t1 = getelementptr double* %p, i32 1
  %x = load double* %t1
  call void @use(double %x)
  call void @use(double %y)
  ret void
}

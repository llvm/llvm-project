; Just check that this completes without hitting an infinite loop.
; RUN: opt < %s --passes=instcombine -S > /dev/null

define i1 @f(i2 %0, i2 %1, i1 %2, i2 %3) {
entry:
  %4 = icmp sgt i2 0, %0
  %5 = zext i1 %2 to i2
  %6 = select i1 %4, i2 0, i2 %5
  %7 = trunc i2 %6 to i1
  %8 = select i1 %7, i2 %1, i2 0
  %9 = icmp sle i1 %4, %7
  %10 = select i1 %9, i2 0, i2 %1
  %11 = and i2 %5, %10
  %12 = select i1 %2, i2 0, i2 %3
  %13 = sdiv i2 1, %12
  %14 = icmp uge i2 %8, %11
  %15 = sext i1 %14 to i2
  %16 = icmp sgt i2 %13, %15
  %17 = icmp sgt i2 %6, 0
  %18 = lshr i1 %16, %17
  ret i1 %18
}

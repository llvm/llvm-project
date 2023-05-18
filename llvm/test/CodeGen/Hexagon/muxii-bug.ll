; RUN: llc -march=hexagon < %s | FileCheck %s

; Make sure "generate mux" pass does not optimize out the value "1908".
; CHECK-LABEL: foo
; CHECK: 1908
define internal fastcc i32 @foo(i32) #0 {
  %2 = icmp eq i32 %0, 1
  %3 = select i1 %2, i32 1712, i32 0
  %4 = icmp eq i32 %0, 1
  %5 = select i1 %4, i32 1908, i32 %3
  %6 = icmp eq i32 %0, 1
  %7 = icmp ult i32 %5, 1740
  %8 = and i1 %6, %7
  %9 = select i1 %8, i32 1740, i32 %5
  %10 = icmp eq i32 %0, 1
  %11 = icmp ult i32 %9, 1732
  %12 = and i1 %10, %11
  %13 = select i1 %12, i32 1732, i32 %9
  %14 = icmp eq i32 %0, 2
  %15 = icmp ult i32 %13, 1936
  %16 = and i1 %14, %15
  %17 = select i1 %16, i32 1936, i32 %13
  %18 = icmp eq i32 %0, 1
  %19 = icmp ult i32 %17, 1580
  %20 = and i1 %18, %19
  %21 = select i1 %20, i32 1580, i32 %17
  ret i32 %21
}

attributes #0 = { nounwind }

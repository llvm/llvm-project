; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {icmp s\[lg\]t i32 %n, 0} | count 16

; Instcombine should recognize that this code can be adjusted
; to fit the canonical smax/smin pattern.

define i32 @floor_a(i32 %n) {
  %t = icmp sgt i32 %n, -1
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @ceil_a(i32 %n) {
  %t = icmp slt i32 %n, 1
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @floor_b(i32 %n) {
  %t = icmp sgt i32 %n, 0
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @ceil_b(i32 %n) {
  %t = icmp slt i32 %n, 0
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @floor_c(i32 %n) {
  %t = icmp sge i32 %n, 0
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @ceil_c(i32 %n) {
  %t = icmp sle i32 %n, 0
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @floor_d(i32 %n) {
  %t = icmp sge i32 %n, 1
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @ceil_d(i32 %n) {
  %t = icmp sle i32 %n, -1
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @floor_e(i32 %n) {
  %t = icmp sgt i32 %n, -1
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @ceil_e(i32 %n) {
  %t = icmp slt i32 %n, 1
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @floor_f(i32 %n) {
  %t = icmp sgt i32 %n, 0
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @ceil_f(i32 %n) {
  %t = icmp slt i32 %n, 0
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @floor_g(i32 %n) {
  %t = icmp sge i32 %n, 0
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @ceil_g(i32 %n) {
  %t = icmp sle i32 %n, 0
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @floor_h(i32 %n) {
  %t = icmp sge i32 %n, 1
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}
define i32 @ceil_h(i32 %n) {
  %t = icmp sle i32 %n, -1
  %m = select i1 %t, i32 %n, i32 0
  ret i32 %m
}

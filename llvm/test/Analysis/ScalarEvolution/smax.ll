; RUN: llvm-as < %s | opt -analyze -scalar-evolution | grep smax | count 2
; RUN: llvm-as < %s | opt -analyze -scalar-evolution | grep \
; RUN:     {%. smax %. smax %.}
; PR1614

define i32 @x(i32 %a, i32 %b, i32 %c) {
  %A = icmp sgt i32 %a, %b
  %B = select i1 %A, i32 %a, i32 %b
  %C = icmp sle i32 %c, %B
  %D = select i1 %C, i32 %B, i32 %c
  ret i32 %D
}

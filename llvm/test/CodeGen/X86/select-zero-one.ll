; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep cmov
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep xor
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movzbl | count 1

@r1 = weak global i32 0

define void @t1(i32 %a, double %b) {
  %tmp114 = fcmp ugt double %b, 1.000000e-09
  %tmp120 = icmp eq i32 %a, 0		; <i1> [#uses=1]
  %bothcond = or i1 %tmp114, %tmp120		; <i1> [#uses=1]
  %storemerge = select i1 %bothcond, i32 0, i32 1		; <i32> [#uses=2]
  store i32 %storemerge, i32* @r1, align 4
  ret void
}

@r2 = weak global i8 0

define void @t2(i32 %a, double %b) {
  %tmp114 = fcmp ugt double %b, 1.000000e-09
  %tmp120 = icmp eq i32 %a, 0		; <i1> [#uses=1]
  %bothcond = or i1 %tmp114, %tmp120		; <i1> [#uses=1]
  %storemerge = select i1 %bothcond, i8 0, i8 1		; <i32> [#uses=2]
  store i8 %storemerge, i8* @r2, align 4
  ret void
}

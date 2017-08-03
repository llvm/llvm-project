; RUN: llc < %s -march=x86 -disable-cgp-branch-opts | grep movzbl
; PR3366

define void @_ada_c34002a() nounwind {
entry:
  %0 = load i8, i8* null, align 1
  %1 = sdiv i8 90, %0
  %2 = icmp ne i8 %1, 3
  %3 = zext i1 %2 to i8
  %toBool449 = icmp ne i8 %3, 0
  %4 = or i1 false, %toBool449
  %5 = zext i1 %4 to i8
  %toBool450 = icmp ne i8 %5, 0
  br i1 %toBool450, label %bb451, label %bb457

bb451:
  br label %bb457

bb457:
  unreachable
}

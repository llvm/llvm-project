; RUN: llvm-as < %s -o /dev/null

define i8 @f1(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !0
  ret i8 %y
}
!0 = metadata !{i8 0, i8 1}

define i8 @f2(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !1
  ret i8 %y
}
!1 = metadata !{i8 255, i8 1}

define i8 @f3(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !2
  ret i8 %y
}
!2 = metadata !{i8 1, i8 3, i8 5, i8 42}

define i8 @f4(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !3
  ret i8 %y
}
!3 = metadata !{i8 -1, i8 0, i8 1, i8 2}

define i8 @f5(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !4
  ret i8 %y
}
!4 = metadata !{i8 -1, i8 0, i8 1, i8 -2}

; RUN: opt %s -strip -S | FileCheck %s
; PR10286

@main_addrs = constant [2 x i8*] [i8* blockaddress(@f, %FOO), i8* blockaddress(@f, %BAR)]
; CHECK: @main_addrs = constant [2 x i8*] [i8* blockaddress(@f, %2), i8* blockaddress(@f, %3)]

declare void @foo() nounwind
declare void @bar() nounwind

define void @f(i8* %indirect.goto.dest) nounwind uwtable ssp {
entry:
  indirectbr i8* %indirect.goto.dest, [label %FOO, label %BAR]

  ; CHECK: indirectbr i8* %0, [label %2, label %3]

FOO:
  call void @foo()
  ret void

BAR:
  call void @bar()
  ret void
}

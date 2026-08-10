; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o /dev/null

declare void @bar(i64 %x, i64 %y)

define void @foo() {
  call void @bar(i64 2, i64 3)
  ret void
}

; RUN: llc < %s -mtriple=x86_64-- -mcpu=corei7

define void @autogen_SD3100() {
BB:
  %FC123 = fptoui float 0x40693F5D00000000 to i1
  br i1 %FC123, label %V, label %W

V:
  ret void
W:
  ret void
}

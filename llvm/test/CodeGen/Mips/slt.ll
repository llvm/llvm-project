; RUN: llc -mtriple=mips -mcpu=mips32r3 -mattr=micromips -relocation-model=pic < %s | FileCheck %s

define i32 @slt(i32 signext %a) nounwind readnone {
  %1 = icmp slt i32 %a, 7
  ; CHECK-LABEL: slt:
  ; CHECK: slt ${{[0-9]+}}, ${{[0-9]+}}, $4
  %2 = select i1 %1, i32 3, i32 4
  ret i32 %2
}

define i32 @sgt(i32 signext %a) {
entry:
  ; CHECK-LABEL: sgt:
  %cmp = icmp sgt i32 %a, 32767
  ; CHECK: slt ${{[0-9]+}}, ${{[0-9]+}}, $4
  %cond = select i1 %cmp, i32 7, i32 5
  ret i32 %cond
}

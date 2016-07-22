; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-eabi | FileCheck %s

; CHECK: mrs [[NZCV_SAVE:x[0-9]+]], NZCV
; CHECK: msr NZCV, [[NZCV_SAVE]]

; DAG ends up with two uses for the flags from an ADCS node, which means they
; must be saved for later.
define void @f(i256* nocapture %a, i256* nocapture %b, i256* nocapture %cc, i256* nocapture %dd) nounwind uwtable noinline ssp {
entry:
  %c = load i256, i256* %cc
  %d = load i256, i256* %dd
  %add = add nsw i256 %c, %d
  store i256 %add, i256* %a, align 8
  %or = or i256 %c, 1606938044258990275541962092341162602522202993782792835301376
  %add6 = add nsw i256 %or, %d
  store i256 %add6, i256* %b, align 8
  ret void
}

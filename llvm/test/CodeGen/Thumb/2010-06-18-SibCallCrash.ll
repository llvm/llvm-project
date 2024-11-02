; RUN: llc -mtriple=thumb-eabi %s -o /dev/null
; rdar://8104457

define arm_apcscc void @t(ptr %m) nounwind {
entry:
  tail call arm_apcscc  void undef(ptr %m, i16 zeroext undef) nounwind
  ret void
}

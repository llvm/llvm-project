; RUN: llc -march=thumb < %s
; rdar://8104457

define arm_apcscc void @t(i32* %m) nounwind {
entry:
  tail call arm_apcscc  void undef(i32* %m, i16 zeroext undef) nounwind
  ret void
}

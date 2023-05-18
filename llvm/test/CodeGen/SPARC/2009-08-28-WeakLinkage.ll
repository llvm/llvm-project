; RUN: llc -march=sparc < %s | grep weak

define weak i32 @func() nounwind {
entry:
  ret i32 0
}

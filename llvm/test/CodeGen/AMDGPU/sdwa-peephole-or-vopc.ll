; RUN: not --crash llc -mtriple=amdgpu10.30 < %s

; VOPC SDWA has no dst_sel, so SIPeepholeSDWA crashes when it tries to fold
; an OR with a VOPC SDWA operand assuming one exists.

define i32 @or_of_vopc_sdwa(i32 %a, i32 %b) {
  %ah = lshr i32 %a, 16
  %bh = lshr i32 %b, 16
  %sum = add i32 %ah, %bh
  %mask = call i32 @llvm.amdgcn.icmp.i32.i32(i32 %ah, i32 %bh, i32 32)
  %r = or i32 %sum, %mask
  ret i32 %r
}

; RUN: llc < %s -mtriple=x86_64-linux > %t
; RUN: grep {movzbl	%\[abcd\]h,} %t | count 8
; RUN: grep {%\[abcd\]h} %t | not grep {%r\[\[:digit:\]\]*d}

; LLVM creates virtual registers for values live across blocks
; based on the type of the value. Make sure that the extracts
; here use the GR64_NOREX register class for their result,
; instead of plain GR64.

define i64 @foo(i64 %a, i64 %b, i64 %c, i64 %d,
                i64 %e, i64 %f, i64 %g, i64 %h) {
  %sa = lshr i64 %a, 8
  %A = and i64 %sa, 255
  %sb = lshr i64 %b, 8
  %B = and i64 %sb, 255
  %sc = lshr i64 %c, 8
  %C = and i64 %sc, 255
  %sd = lshr i64 %d, 8
  %D = and i64 %sd, 255
  %se = lshr i64 %e, 8
  %E = and i64 %se, 255
  %sf = lshr i64 %f, 8
  %F = and i64 %sf, 255
  %sg = lshr i64 %g, 8
  %G = and i64 %sg, 255
  %sh = lshr i64 %h, 8
  %H = and i64 %sh, 255
  br label %next

next:
  %u = add i64 %A, %B
  %v = add i64 %C, %D
  %w = add i64 %E, %F
  %x = add i64 %G, %H
  %y = add i64 %u, %v
  %z = add i64 %w, %x
  %t = add i64 %y, %z
  ret i64 %t
}

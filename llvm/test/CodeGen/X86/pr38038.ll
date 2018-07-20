; RUN: llc < %s
; PR38038

define i8 @crash(half)  {
entry:
  %1 = bitcast half %0 to i16
  %.lobit = lshr i16 %1, 15
  %2 = trunc i16 %.lobit to i8
  ret i8 %2
}

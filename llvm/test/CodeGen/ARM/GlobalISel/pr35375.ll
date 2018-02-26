; RUN: llc -O0 -mtriple armv7-- -stop-before=expand-isel-pseudos < %s
; RUN: llc -O0 -mtriple armv7-- -stop-before=expand-isel-pseudos -global-isel < %s

; CHECK: PKHBT

define arm_aapcscc i32 @pkh(i32 %x, i32 %y) {
  %andx = and i32 %x, 65535
  %shl = shl i32 %y, 1
  %andy = and i32 %shl, 4294901760 ; same as -65536
  %or = or i32 %andx, %andy
  ret i32 %or
}

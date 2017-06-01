; RUN: llc %s -O0 -march=sparc -mcpu=leon2 -o - | FileCheck %s
; RUN: llc %s -O0 -march=sparc -mcpu=leon3 -o - | FileCheck %s
; RUN: llc %s -O0 -march=sparc -mcpu=leon4 -o - | FileCheck %s

; CHECK-LABEL: smac_test:
; CHECK:       smac %o1, %o0, %o0
define i32 @smac_test(i16* %a, i16* %b) {
entry:
;  %0 = tail call i32 asm sideeffect "smac $2, $1, $0", "={r2},{r3},{r4}"(i16* %a, i16* %b)
  %0 = tail call i32 asm sideeffect "smac $2, $1, $0", "=r,rI,r"(i16* %a, i16* %b)
  ret i32 %0
}

; CHECK-LABEL: umac_test:
; CHECK:       umac %o1, %o0, %o0
define i32 @umac_test(i16* %a, i16* %b) {
entry:
  %0 = tail call i32 asm sideeffect "umac $2, $1, $0", "=r,rI,r"(i16* %a, i16* %b)
  ret i32 %0
}

; RUN: llc %s -O0 -mtriple=sparc -mcpu=leon2 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=leon3 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=leon4 -o - | FileCheck %s

; CHECK-LABEL: smac_test:
; CHECK:       smac %i1, %i0, %i0
define i32 @smac_test(ptr %a, ptr %b) {
entry:
;  %0 = tail call i32 asm sideeffect "smac $2, $1, $0", "={r2},{r3},{r4}"(i16* %a, i16* %b)
  %0 = tail call i32 asm sideeffect "smac $2, $1, $0", "=r,rI,r"(ptr %a, ptr %b)
  ret i32 %0
}

; CHECK-LABEL: umac_test:
; CHECK:       umac %i1, %i0, %i0
define i32 @umac_test(ptr %a, ptr %b) {
entry:
  %0 = tail call i32 asm sideeffect "umac $2, $1, $0", "=r,rI,r"(ptr %a, ptr %b)
  ret i32 %0
}

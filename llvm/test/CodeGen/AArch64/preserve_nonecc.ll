; RUN: sed -e "s/RETTYPE/void/;s/RETVAL//" %s | llc -mtriple=aarch64-apple-darwin | FileCheck --check-prefixes=ALL %s
; RUN: sed -e "s/RETTYPE/i32/;s/RETVAL/undef/" %s | llc -mtriple=aarch64-apple-darwin | FileCheck --check-prefixes=ALL %s
; RUN: sed -e "s/RETTYPE/\{i64\,i64\}/;s/RETVAL/undef/" %s | llc -mtriple=aarch64-apple-darwin | FileCheck --check-prefixes=ALL %s
; RUN: sed -e "s/RETTYPE/double/;s/RETVAL/0./" %s | llc -mtriple=aarch64-apple-darwin | FileCheck --check-prefixes=ALL,DOUBLE %s

; We don't need to save registers before using them inside preserve_none function.
define preserve_nonecc RETTYPE @preserve_nonecc1(i64, i64, double, double) nounwind {
entry:
;ALL-LABEL:   preserve_nonecc1
;ALL:         ; %bb.0:
;ALL-NEXT:    InlineAsm Start
;ALL-NEXT:    InlineAsm End
;DOUBLE-NEXT: movi d0, #0000000000000000
;ALL-NEXT:    ret
  call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{d0},~{d1},~{d2},~{d3},~{d4},~{d5},~{d6},~{d7},~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15},~{d16}"()
  ret RETTYPE RETVAL
}

; When calling a preserve_none function, all live registers must be saved and
; restored around the function call.
declare preserve_nonecc RETTYPE @preserve_nonecc2(i64, i64, double, double)
define void @bar() nounwind {
entry:
;ALL-LABEL: bar
;ALL:       InlineAsm Start
;ALL:       stp x9, x8
;ALL:       stp x11, x10
;ALL:       stp x13, x12
;ALL:       stp x15, x14
;ALL:       stp x17, x16
;ALL:       stp x20, x19
;ALL:       stp x22, x21
;ALL:       stp x24, x23
;ALL:       stp x26, x25
;ALL:       stp x28, x27
;ALL:       stp d8, d7
;ALL:       stp d10, d9
;ALL:       stp d12, d11
;ALL:       stp d14, d13
;ALL:       stp d16, d15
;ALL:       ldp x20, x19
;ALL:       ldp x22, x21
;ALL:       ldp x24, x23
;ALL:       ldp x26, x25
;ALL:       ldp x28, x27
;ALL:       ldp d8, d7
;ALL:       ldp d10, d9
;ALL:       ldp d12, d11
;ALL:       ldp d14, d13
;ALL:       ldp d16, d15
;ALL:       ldp x9, x8
;ALL:       ldp x11, x10
;ALL:       ldp x13, x12
;ALL:       ldp x15, x14
;ALL:       ldp x17, x16
;ALL:       InlineAsm Start
  %a0 = call i64 asm sideeffect "", "={x8}"() nounwind
  %a1 = call i64 asm sideeffect "", "={x9}"() nounwind
  %a2 = call i64 asm sideeffect "", "={x10}"() nounwind
  %a3 = call i64 asm sideeffect "", "={x11}"() nounwind
  %a4 = call i64 asm sideeffect "", "={x12}"() nounwind
  %a5 = call i64 asm sideeffect "", "={x13}"() nounwind
  %a6 = call i64 asm sideeffect "", "={x14}"() nounwind
  %a7 = call i64 asm sideeffect "", "={x15}"() nounwind
  %a8 = call i64 asm sideeffect "", "={x16}"() nounwind
  %a9 = call i64 asm sideeffect "", "={x17}"() nounwind
  %a10 = call i64 asm sideeffect "", "={x19}"() nounwind
  %a11 = call i64 asm sideeffect "", "={x20}"() nounwind
  %a12 = call i64 asm sideeffect "", "={x21}"() nounwind
  %a13 = call i64 asm sideeffect "", "={x22}"() nounwind
  %a14 = call i64 asm sideeffect "", "={x23}"() nounwind
  %a15 = call i64 asm sideeffect "", "={x24}"() nounwind
  %a16 = call i64 asm sideeffect "", "={x25}"() nounwind
  %a17 = call i64 asm sideeffect "", "={x26}"() nounwind
  %a18 = call i64 asm sideeffect "", "={x27}"() nounwind
  %a19 = call i64 asm sideeffect "", "={x28}"() nounwind

  %f0 = call <1 x double> asm sideeffect "", "={d7}"() nounwind
  %f1 = call <1 x double> asm sideeffect "", "={d8}"() nounwind
  %f2 = call <1 x double> asm sideeffect "", "={d9}"() nounwind
  %f3 = call <1 x double> asm sideeffect "", "={d10}"() nounwind
  %f4 = call <1 x double> asm sideeffect "", "={d11}"() nounwind
  %f5 = call <1 x double> asm sideeffect "", "={d12}"() nounwind
  %f6 = call <1 x double> asm sideeffect "", "={d13}"() nounwind
  %f7 = call <1 x double> asm sideeffect "", "={d14}"() nounwind
  %f8 = call <1 x double> asm sideeffect "", "={d15}"() nounwind
  %f9 = call <1 x double> asm sideeffect "", "={d16}"() nounwind

  call preserve_nonecc RETTYPE @preserve_nonecc2(i64 1, i64 2, double 3.0, double 4.0)
  call void asm sideeffect "", "{x8},{x9},{x10},{x11},{x12},{x13},{x14},{x15},{x16},{x17},{x19},{x20},{x21},{x22},{x23},{x24},{x25},{x26},{x27},{x28},{d7},{d8},{d9},{d10},{d11},{d12},{d13},{d14},{d15},{d16}"(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %a12, i64 %a13, i64 %a14, i64 %a15, i64 %a16, i64 %a17, i64 %a18, i64 %a19, <1 x double> %f0, <1 x double> %f1, <1 x double> %f2, <1 x double> %f3, <1 x double> %f4, <1 x double> %f5, <1 x double> %f6, <1 x double> %f7, <1 x double> %f8, <1 x double> %f9)
  ret void
}

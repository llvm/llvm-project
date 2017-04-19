; RUN: llc < %s -O2 -mtriple=x86_64-linux-android -mattr=+mmx \
; RUN:     -enable-legalize-types-checking | FileCheck %s
; RUN: llc < %s -O2 -mtriple=x86_64-linux-gnu -mattr=+mmx \
; RUN:     -enable-legalize-types-checking | FileCheck %s

; Test the softened result of extractelement op code.
define fp128 @TestExtract(<2 x double> %x) {
entry:
  ; Simplified instruction pattern from the output of llvm before r289042,
  ; for a boost function ...::insert<...>::traverse<...>().
  %a = fpext <2 x double> %x to <2 x fp128>
  %0 = extractelement <2 x fp128> %a, i32 0
  %1 = extractelement <2 x fp128> %a, i32 1
  %2 = fmul fp128 %0, %1
  ret fp128 %2
; CHECK-LABEL: TestExtract:
; CHECK:       movaps	%xmm0, (%rsp)
; CHECK:       callq	__extenddftf2
; CHECK:       callq	__extenddftf2
; CHECK:       callq    __multf3
; CHECK:       retq
}

; Test the ADA section in the assembly output for all cases.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

; CHECK-LABEL: DoIt:
; CHECK:    stmg    6, 7, 1840(4)
; CHECK:    aghi    4, -224
; CHECK:    lg  1, 0(5)
; CHECK:    lg  6, 16(5)
; CHECK:    lg  5, 8(5)
; CHECK:    stg 1, 2264(4)
; CHECK:    basr    7, 6
; CHECK:    bcr 0, 0
; CHECK:    lg  7, 2072(4)
; CHECK:    aghi    4, 224
; CHECK:    b   2(7)
define hidden void @DoIt() {
entry:
  %F = alloca ptr, align 8
  store ptr @DoFunc, ptr %F, align 8
  %0 = load ptr, ptr %F, align 8
  call void @Caller(ptr noundef %0)
  ret void
}
declare void @DoFunc()
declare void @Caller(ptr noundef)

; CHECK-LABEL: get_i:
; CHECK:    stmg    6, 8, 1872(4)
; CHECK:    aghi    4, -192
; CHECK:    lg  1, 24(5)
; CHECK:    lg  2, 32(5)
; CHECK:    lgf 1, 0(1)
; CHECK:    lg  6, 48(5)
; CHECK:    lg  5, 40(5)
; CHECK:    l   8, 0(2)
; CHECK:    basr    7, 6
; CHECK:    bcr 0, 0
; CHECK:    ar  3, 8
; CHECK:    lgfr    3, 3
; CHECK:    lmg 7, 8, 2072(4)
; CHECK:    aghi    4, 192
; CHECK:    b   2(7)
@i = external global i32, align 4
@i2 = external global i32, align 4

define signext i32 @get_i() {
entry:
  %0 = load i32, ptr @i, align 4
  %1 = load i32, ptr @i2, align 4
  %call = call signext i32 @callout(i32 signext %1)
  %add = add nsw i32 %0, %call
  ret i32 %add
}

declare signext i32 @callout(i32 signext)

; CHECK:     .section    ".ada"
; CHECK:  .set L#DoFunc@indirect0, DoFunc
; CHECK:      .indirect_symbol   L#DoFunc@indirect0
; CHECK:  .quad V(L#DoFunc@indirect0)          * Offset 0 pointer to function descriptor DoFunc
; CHECK:  .quad R(Caller)                      * Offset 8 function descriptor of Caller
; CHECK:  .quad V(Caller)
; CHECK:  .quad A(i2)                           * Offset 24 pointer to data symbol i2
; CHECK:  .quad A(i)                            * Offset 32 pointer to data symbol i
; CHECK:  .quad R(callout)                      * Offset 40 function descriptor of callout
; CHECK:  .quad V(callout)

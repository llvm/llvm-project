; RUN: llc < %s -mtriple=arm64-apple-ios | FileCheck %s --check-prefix=CHECK-IOS
; RUN: llc < %s -mtriple=arm64-apple-ios -global-isel | FileCheck %s --check-prefix=CHECK-IOS
; RUN: llc < %s -mtriple=arm64-linux-gnu | FileCheck %s --check-prefix=CHECK-LINUX
; RUN: llc < %s -mtriple=arm64-linux-gnu -code-model=large| FileCheck %s --check-prefix=CHECK-LARGE

; rdar://9188695

define i64 @t() nounwind ssp {
entry:
; CHECK-IOS: lCPI0_0:
; CHECK-IOS:     .quad Ltmp0
; CHECK-IOS-LABEL: _t:
; CHECK-IOS: adrp x[[TMP:[0-9]+]], lCPI0_0@PAGE
; CHECK-IOS: ldr {{x[0-9]+}}, [x[[TMP]], lCPI0_0@PAGEOFF]

; CHECK-LINUX-LABEL: t:
; CHECK-LINUX: adrp [[REG:x[0-9]+]], .Ltmp0
; CHECK-LINUX: add {{x[0-9]+}}, [[REG]], :lo12:.Ltmp0

; CHECK-LARGE-LABEL: t:
; CHECK-LARGE: movz [[ADDR_REG:x[0-9]+]], #:abs_g0_nc:[[DEST_LBL:.Ltmp[0-9]+]]
; CHECK-LARGE: movk [[ADDR_REG]], #:abs_g1_nc:[[DEST_LBL]]
; CHECK-LARGE: movk [[ADDR_REG]], #:abs_g2_nc:[[DEST_LBL]]
; CHECK-LARGE: movk [[ADDR_REG]], #:abs_g3:[[DEST_LBL]]

  %recover = alloca i64, align 8
  store volatile i64 ptrtoint (i8* blockaddress(@t, %mylabel) to i64), i64* %recover, align 8
  br label %mylabel

mylabel:
  %tmp = load volatile i64, i64* %recover, align 8
  ret i64 %tmp
}

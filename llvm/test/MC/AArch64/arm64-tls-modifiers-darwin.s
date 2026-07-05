; RUN: llvm-mc -triple=arm64-apple-ios7.0 %s -o - | FileCheck %s

        adrp x2, _var@TLVPPAGE
        ldr x0, [x15, _var@TLVPPAGEOFF]
        add x30, x0, _var@TLVPPAGEOFF
; CHECK: adrp x2, _var@TLVPPAG
; CHECK: ldr x0, [x15, _var@TLVPPAGEOFF]
; CHECK: add x30, x0, _var@TLVPPAGEOFF

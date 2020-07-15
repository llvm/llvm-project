; RUN: llc -mtriple=arm64e-apple-ios %s -o - | FileCheck %s

@var = thread_local global i8 0

define i8 @get_var() #0 {
; CHECK-LABEL: get_var:
; CHECK: adrp x[[TLVPDESC_SLOT_HI:[0-9]+]], _var@TLVPPAGE
; CHECK: ldr x[[PTR:[0-9]+]], [x[[TLVPDESC_SLOT_HI]], _var@TLVPPAGEOFF]
; CHECK: ldr [[TLV_GET_ADDR:x[0-9]+]], [x[[PTR]]]
; CHECK: blraaz [[TLV_GET_ADDR]]
; CHECK: ldrb w0, [x0]

  %val = load i8, i8* @var, align 1
  ret i8 %val
}

attributes #0 = { nounwind "ptrauth-calls" }

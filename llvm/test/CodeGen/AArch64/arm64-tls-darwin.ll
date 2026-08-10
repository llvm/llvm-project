; RUN: llc -mtriple=arm64-apple-ios7.0 %s -o - | FileCheck %s
; RUN: llc -mtriple=arm64-apple-ios7.0 -global-isel -global-isel-abort=1 -verify-machineinstrs %s -o - | FileCheck %s

@var = thread_local global i8 0

; N.b. x0 must be the result of the first load (i.e. the address of the
; descriptor) when tlv_get_addr is called. Likewise the result is returned in
; x0.
define i8 @get_var() {
; CHECK-LABEL: get_var:
; CHECK: adrp x[[TLVPDESC_SLOT_HI:[0-9]+]], _var@TLVPPAGE
 ; CHECK: ldr x[[PTR:[0-9]+]], [x[[TLVPDESC_SLOT_HI]], _var@TLVPPAGEOFF]
 ; CHECK: ldr [[TLV_GET_ADDR:x[0-9]+]], [x[[PTR]]]
; CHECK: blr [[TLV_GET_ADDR]]
; CHECK: ldrb w0, [x0]

  %val = load i8, ptr @var, align 1
  ret i8 %val
}

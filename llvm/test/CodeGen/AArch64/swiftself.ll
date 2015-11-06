; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-apple-ios | FileCheck --check-prefix=CHECK-APPLE %s
; RUN: llc -O0 -verify-machineinstrs < %s -mtriple=aarch64-apple-ios | FileCheck --check-prefix=CHECK-O0 %s

; Parameter with swiftself should be allocated to x9.
define void @check_swiftself(i32* swiftself %addr0) {
; CHECK-APPLE-LABEL: check_swiftself:
; CHECK-O0-LABEL: check_swiftself:

    %val0 = load volatile i32, i32* %addr0
; CHECK-APPLE: ldr w{{.*}}, [x9]
; CHECK-O0: ldr w{{.*}}, [x9]
    ret void
}

@var8_3 = global i8 0
declare void @take_swiftself(i8* swiftself %addr0)

define void @simple_args() {
; CHECK-APPLE-LABEL: simple_args:
; CHECK-O0-LABEL: simple_args:

  call void @take_swiftself(i8* @var8_3)
; CHECK-APPLE: add x9,
; CHECK-APPLE: bl {{_?}}take_swiftself
; CHECK-O0: add x9,
; CHECK-O0: bl {{_?}}take_swiftself

  ret void
}

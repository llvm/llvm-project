; RUN: llc %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s
; RUN: llc %s -filetype=asm -o - | FileCheck --check-prefix=ASM %s

target triple = "x86_64-apple-driverkit19.0.0"

define void @foo() {
entry:
  ret void
}

; CHECK:            cmd LC_BUILD_VERSION
; CHECK-NEXT:   cmdsize 24
; CHECK-NEXT:  platform driverkit
; CHECK-NEXT:       sdk n/a
; CHECK-NEXT:     minos 19.0
; CHECK-NEXT:    ntools 0

; ASM: .build_version driverkit, 19, 0

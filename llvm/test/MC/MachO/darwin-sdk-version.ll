; RUN: llc %s -filetype=obj -o - | llvm-objdump -macho -private-headers - | FileCheck %s
; RUN: llc %s -filetype=asm -o - | FileCheck --check-prefix=ASM %s

target triple = "x86_64-apple-macos10.14";
!llvm.module.flags = !{!0};
!0 = !{i32 2, !"SDK Version", [3 x i32] [ i32 10, i32 14, i32 2 ] };

define void @foo() {
entry:
  ret void
}

; CHECK:           cmd LC_VERSION_MIN_MACOSX
; CHECK-NEXT:  cmdsize 16
; CHECK-NEXT:  version 10.14
; CHECK-NEXT:      sdk 10.14.2

; ASM: .macosx_version_min 10, 14 sdk_version 10, 14, 2

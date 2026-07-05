; RUN: llc %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s
; RUN: llc %s -filetype=asm -o - | FileCheck --check-prefix=ASM %s

target triple = "x86_64-apple-macos10.15";
!llvm.module.flags = !{!0, !1, !2};
!0 = !{i32 2, !"SDK Version", [3 x i32] [ i32 10, i32 15, i32 1 ] };
!1 = !{i32 2, !"darwin.target_variant.triple", !"x86_64-apple-ios13.1-macabi"};
!2 = !{i32 2, !"darwin.target_variant.SDK Version", [2 x i32] [ i32 13, i32 2 ] };

define void @foo() {
entry:
  ret void
}

; CHECK:           cmd LC_BUILD_VERSION
; CHECK-NEXT:  cmdsize 24
; CHECK-NEXT: platform macos
; CHECK-NEXT:      sdk 10.15.1
; CHECK-NEXT:    minos 10.15
; CHECK-NEXT:   ntools 0
; CHECK:           cmd LC_BUILD_VERSION
; CHECK-NEXT:  cmdsize 24
; CHECK-NEXT: platform macCatalyst
; CHECK-NEXT:      sdk 13.2
; CHECK-NEXT:    minos 13.1
; CHECK-NEXT:   ntools 0

; ASM: .build_version macos, 10, 15    sdk_version 10, 15, 1
; ASM: .build_version macCatalyst, 13, 1    sdk_version 13, 2

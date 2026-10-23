; RUN: split-file %s %t
; RUN: llc -mattr=+multivalue < %t/mv.ll | FileCheck %s --check-prefix=MV
; RUN: llc -mattr=+multivalue < %t/mvp.ll | FileCheck %s --check-prefix=MVP

; Test that the ABI is selected from the "target-abi" module flag: the
; multivalue return is lowered directly only for "experimental-mv".

;--- mv.ll
target triple = "wasm32-unknown-unknown"

%pair = type { i32, i64 }

; MV: .functype pair_const () -> (i32, i64)
define %pair @pair_const() {
  ret %pair { i32 42, i64 42 }
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"experimental-mv"}

;--- mvp.ll
target triple = "wasm32-unknown-unknown"

%pair = type { i32, i64 }

; MVP: .functype pair_const (i32) -> ()
define %pair @pair_const() {
  ret %pair { i32 42, i64 42 }
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"mvp"}

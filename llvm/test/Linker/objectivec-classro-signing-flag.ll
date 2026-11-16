; Bitcode from before "Objective-C Enforce ClassRO Pointer Signing" existed does
; not sign class_ro_t pointers. It is upgraded to an Override flag of 0, so
; linking it with bitcode that signs them does not keep 16.

; RUN: rm -rf %t && split-file %s %t
; RUN: opt -module-summary %t/old.ll -o %t/old.bc
; RUN: opt -module-summary %t/signed.ll -o %t/signed.bc
; RUN: llvm-as %t/signed2.ll -o %t/signed2.bc
; RUN: llvm-as %t/unsigned.ll -o %t/unsigned.bc

; RUN: llvm-link %t/old.bc %t/signed.bc -S | FileCheck %s
; RUN: llvm-link %t/signed.bc %t/old.bc -S | FileCheck %s
; RUN: llvm-link %t/old.bc %t/unsigned.bc -S | FileCheck %s
; RUN: llvm-link %t/unsigned.bc %t/old.bc -S | FileCheck %s

;; ThinLTO function import merges module flags too.
; RUN: llvm-lto -thinlto -o %t/index %t/old.bc %t/signed.bc
; RUN: llvm-link %t/old.bc -import=f:%t/signed.bc \
; RUN:   -summary-index=%t/index.thinlto.bc -S \
; RUN:   | FileCheck %s --check-prefixes=CHECK,IMPORT

; IMPORT: define available_externally void @f()
; CHECK: !{{[0-9]+}} = !{i32 4, !"Objective-C Enforce ClassRO Pointer Signing", i32 0}

;; Bitcode that has the flag keeps it.
; RUN: llvm-link %t/signed.bc %t/signed2.bc -S \
; RUN:   | FileCheck %s --check-prefix=SIGNED

; SIGNED: !{{[0-9]+}} = !{i32 1, !"Objective-C Enforce ClassRO Pointer Signing", i32 16}

;--- old.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"Objective-C Image Info Version", i32 0}

;--- signed.ll
define void @f() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!1 = !{i32 1, !"Objective-C Enforce ClassRO Pointer Signing", i32 16}

;--- signed2.ll
!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!1 = !{i32 1, !"Objective-C Enforce ClassRO Pointer Signing", i32 16}

;--- unsigned.ll
!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!1 = !{i32 1, !"Objective-C Enforce ClassRO Pointer Signing", i8 0}

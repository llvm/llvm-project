;; The PowerPC "float-abi" module flag (describing the long double
;; format) was renamed to "long-double-type", with its values changed
;; to the corresponding IR floating-point type names.

; RUN: split-file %s %t
; RUN: llvm-as < %t/doubledouble.ll | llvm-dis | FileCheck %s --check-prefix=DOUBLEDOUBLE
; RUN: llvm-as < %t/ieeequad.ll | llvm-dis | FileCheck %s --check-prefix=IEEEQUAD
; RUN: llvm-as < %t/ieeedouble.ll | llvm-dis | FileCheck %s --check-prefix=IEEEDOUBLE
; RUN: llvm-as < %t/unrecognized.ll | llvm-dis | FileCheck %s --check-prefix=UNRECOGNIZED
; RUN: llvm-as < %t/arm.ll | llvm-dis | FileCheck %s --check-prefix=ARM

;; All the old PowerPC long double format spellings are upgraded to the new key
;; and IR type-name values.
;--- doubledouble.ll
target triple = "powerpc64le-unknown-linux-gnu"
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"doubledouble"}
; DOUBLEDOUBLE: !0 = !{i32 1, !"long-double-type", !"ppc_fp128"}

;--- ieeequad.ll
target triple = "powerpc64le-unknown-linux-gnu"
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"ieeequad"}
; IEEEQUAD: !0 = !{i32 1, !"long-double-type", !"fp128"}

;--- ieeedouble.ll
target triple = "powerpc64le-unknown-linux-gnu"
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"ieeedouble"}
; IEEEDOUBLE: !0 = !{i32 1, !"long-double-type", !"double"}

;--- unrecognized.ll
;; An unrecognized value was never valid; upgrade it to the PowerPC default
;; ppc_fp128 rather than leaving a stale "float-abi" flag.
target triple = "powerpc64le-unknown-linux-gnu"
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"bogus"}
; UNRECOGNIZED: !0 = !{i32 1, !"long-double-type", !"ppc_fp128"}

;--- arm.ll
;; A non-PowerPC module is never touched.
target triple = "armv7-unknown-linux-gnueabihf"
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"hard"}
; ARM: !0 = !{i32 1, !"float-abi", !"hard"}

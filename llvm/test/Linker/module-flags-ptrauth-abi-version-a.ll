; RUN: llvm-link %p/Inputs/module-flags-ptrauth-abi-version-5.ll %p/Inputs/module-flags-ptrauth-abi-version-5.ll -S -o - | FileCheck %s --check-prefix=SAME --check-prefix=CHECK
; RUN: llvm-link %p/Inputs/module-flags-ptrauth-abi-version-3.ll %p/Inputs/module-flags-ptrauth-abi-version-5.ll -S -o - | FileCheck %s --check-prefix=DIFFERENT --check-prefix=CHECK
; RUN: llvm-link %p/Inputs/module-flags-ptrauth-abi-version-3.ll %p/Inputs/module-flags-ptrauth-abi-version-5.ll %p/Inputs/module-flags-ptrauth-abi-version-3.ll -S -o - | FileCheck %s --check-prefix=DIFFERENT-MORE --check-prefix=CHECK
; RUN: llvm-link %s %p/Inputs/module-flags-ptrauth-abi-version-3.ll -S -o - | FileCheck %s --check-prefix=EMPTY --check-prefix=CHECK


; CHECK: !llvm.module.flags = !{!0}
; CHECK: !0 = {{(distinct )?}}!{i32 6, !"ptrauth.abi-version", !1}

; test linking modules with the same ptrauth.abi-version: it should merge them
; SAME: !1 = distinct !{!2}
; SAME: !2 = !{i32 5, i1 false}

; test linking modules with different ptrauth.abi-versions: it should append them
; DIFFERENT: !1 = distinct !{!2, !3}
; DIFFERENT: !2 = !{i32 3, i1 false}
; DIFFERENT: !3 = !{i32 5, i1 false}

; test linking modules with three different ptrauth.abi-versions where two are the same: it should unique and append them
; DIFFERENT-MORE: !1 = distinct !{!2, !3}
; DIFFERENT-MORE: !2 = !{i32 3, i1 false}
; DIFFERENT-MORE: !3 = !{i32 5, i1 false}

; test linking modules with no ptrauth.abi-version on one side: it should pick it up from the one that has it
; EMPTY: !1 = !{!2}
; EMPTY: !2 = !{i32 3, i1 false}

; FIXME: test linking modules with no ptrauth.abi-version on one side: we
; auto-upgrade it to the version -1, and append it. We can't do this now because
; the module materializer doesn't run the auto-upgrader during llvm-link.

; test linking modules with different modes: kernel and userand user: it should append them
; DIFFERENT-KERNEL: !1 = !{!2, !3}
; DIFFERENT-KERNEL: !2 = !{i32 3, i1 true}
; DIFFERENT-KERNEL: !3 = !{i32 3, i1 false}

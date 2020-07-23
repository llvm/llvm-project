;; Test upgrade linker option doesn't create duplicated linker options.
;; Inputs is generated from IR and checked in as bitcode as it will get rejected by verifier.
;; define void @test() {
;;   ret void
;; }
;; !llvm.module.flags = !{!0}
;; !0 = !{i32 6, !"Linker Options", !1}
;; !1 = !{!2}
;; !2 = !{!"-framework", !"Foundation"}

; RUN: llvm-dis %S/Inputs/linker-options.bc -o - | FileCheck %s
; CHECK: !llvm.linker.options = !{!2}

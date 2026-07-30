; RUN: split-file %s %t
; RUN: llvm-as < %t/dwarf.ll | llvm-dis | FileCheck %s --check-prefix=DWARF
; RUN: llvm-as < %t/sjlj.ll | llvm-dis | FileCheck %s --check-prefix=SJLJ
; RUN: llvm-as < %t/arm.ll | llvm-dis | FileCheck %s --check-prefix=ARM
; RUN: llvm-as < %t/wineh.ll | llvm-dis | FileCheck %s --check-prefix=WINEH
; RUN: llvm-as < %t/wasm.ll | llvm-dis | FileCheck %s --check-prefix=WASM

;--- dwarf.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"dwarf"}
; DWARF: !0 = !{i32 1, !"exception-model", !"dwarf"}

;--- sjlj.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"sjlj"}
; SJLJ: !0 = !{i32 1, !"exception-model", !"sjlj"}

;--- arm.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"arm"}
; ARM: !0 = !{i32 1, !"exception-model", !"arm"}

;--- wineh.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"wineh"}
; WINEH: !0 = !{i32 1, !"exception-model", !"wineh"}

;--- wasm.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"wasm"}
; WASM: !0 = !{i32 1, !"exception-model", !"wasm"}

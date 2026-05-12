; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Make sure that the ModuleNameHash flag is successfully serialized to bitcode
; and deserialized back.

!llvm.module.flags = !{!0}

!0 = !{i32 4, !"ModuleNameHash", !"323802289588745424661866488133278119720"}

; CHECK: !llvm.module.flags = !{!0}
; CHECK: !0 = !{i32 4, !"ModuleNameHash", !"323802289588745424661866488133278119720"}

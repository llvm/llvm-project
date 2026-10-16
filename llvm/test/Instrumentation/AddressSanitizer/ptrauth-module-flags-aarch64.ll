; REQUIRES: aarch64-registered-target
; RUN: rm -rf %t && split-file %s %t && cd %t

; RUN: opt -mtriple=aarch64-linux-gnu < all.ll -passes=asan -S | FileCheck %s --check-prefix=ALL

; RUN: opt -mtriple=aarch64-linux-gnu < returns.ll -passes=asan -S | FileCheck %s --check-prefix=RET \
; RUN:   --implicit-check-not=ptrauth-auth-traps --implicit-check-not=ptrauth-indirect-gotos --implicit-check-not=aarch64-jump-table-hardening

; RUN: opt -mtriple=aarch64-linux-gnu < auth-traps.ll -passes=asan -S | FileCheck %s --check-prefix=TRAP \
; RUN:   --implicit-check-not=ptrauth-returns --implicit-check-not=ptrauth-indirect-gotos --implicit-check-not=aarch64-jump-table-hardening

; RUN: opt -mtriple=aarch64-linux-gnu < indirect-gotos.ll -passes=asan -S | FileCheck %s --check-prefix=GOTO \
; RUN:   --implicit-check-not=ptrauth-returns --implicit-check-not=ptrauth-auth-traps --implicit-check-not=aarch64-jump-table-hardening

; RUN: opt -mtriple=aarch64-linux-gnu < jump-table-hardening.ll -passes=asan -S | FileCheck %s --check-prefix=JUMP \
; RUN:   --implicit-check-not=ptrauth-returns --implicit-check-not=ptrauth-auth-traps --implicit-check-not=ptrauth-indirect-gotos

;--- all.ll

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 7, !"ptrauth-returns", i32 1}
!1 = !{i32 7, !"ptrauth-auth-traps", i32 1}
!2 = !{i32 7, !"ptrauth-indirect-gotos", i32 1}
!3 = !{i32 7, !"aarch64-jump-table-hardening", i32 1}

; ALL: define internal void @asan.module_ctor() #[[#ATTR:]]
; ALL: attributes #[[#ATTR]] = { nounwind "aarch64-jump-table-hardening" "ptrauth-auth-traps" "ptrauth-indirect-gotos" "ptrauth-returns" }

;--- returns.ll

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"ptrauth-returns", i32 1}

; RET: define internal void @asan.module_ctor() #[[#ATTR:]]
; RET: attributes #[[#ATTR]] = { nounwind "ptrauth-returns" }

;--- auth-traps.ll

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"ptrauth-auth-traps", i32 1}

; TRAP: define internal void @asan.module_ctor() #[[#ATTR:]]
; TRAP: attributes #[[#ATTR]] = { nounwind "ptrauth-auth-traps" }

;--- indirect-gotos.ll

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"ptrauth-indirect-gotos", i32 1}

; GOTO: define internal void @asan.module_ctor() #[[#ATTR:]]
; GOTO: attributes #[[#ATTR]] = { nounwind "ptrauth-indirect-gotos" }

;--- jump-table-hardening.ll

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"aarch64-jump-table-hardening", i32 1}

; JUMP: define internal void @asan.module_ctor() #[[#ATTR:]]
; JUMP: attributes #[[#ATTR]] = { nounwind "aarch64-jump-table-hardening" }

; RUN: llc < %s -mtriple=arm64ec-pc-windows-msvc | FileCheck %s

declare void @called()
declare void @escaped()
define void @f(ptr %dst, ptr readonly %f) {
  call void @called()
; CHECK:         bl      "#called"
  store ptr @escaped, ptr %dst
  call void %f()
; CHECK:       adrp    x10, $iexit_thunk$cdecl$v$v
; CHECK-NEXT:  add     x10, x10, :lo12:$iexit_thunk$cdecl$v$v
; CHECK-NEXT:  str     x8, [x20]
; CHECK-NEXT:  adrp    x8, __os_arm64x_check_icall_cfg
; CHECK-NEXT:  ldr     x8, [x8, :lo12:__os_arm64x_check_icall_cfg]
; CHECK-NEXT:  mov     x11,
; CHECK-NEXT:  blr     x8
; CHECK-NEXT:  blr     x11
    ret void
}

; CHECK-LABEL:    .def "#called$exit_thunk";
; CHECK-NEXT:     .scl 2;
; CHECK-NEXT:     .type 32;
; CHECK-NEXT:     .endef
; CHECK-NEXT:     .section .wowthk$aa,"xr",discard,"#called$exit_thunk"
; CHECK-NEXT:     .globl "#called$exit_thunk"            // -- Begin function #called$exit_thunk
; CHECK-NEXT:     .p2align 2
; CHECK-NEXT: "#called$exit_thunk":                   // @"#called$exit_thunk"
; CHECK-NEXT:     .weak_anti_dep called
; CHECK-NEXT: called = "#called"
; CHECK-NEXT:     .weak_anti_dep "#called"
; CHECK-NEXT: "#called" = "#called$exit_thunk"
; CHECK-NEXT:    .seh_proc "#called$exit_thunk"
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:     str x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp x8, __os_arm64x_check_icall
; CHECK-NEXT:     adrp x11, called
; CHECK-NEXT:     add x11, x11, :lo12:called
; CHECK-NEXT:     ldr x8, [x8, :lo12:__os_arm64x_check_icall]
; CHECK-NEXT:     adrp x10, $iexit_thunk$cdecl$v$v
; CHECK-NEXT:     add x10, x10, :lo12:$iexit_thunk$cdecl$v$v
; CHECK-NEXT:     blr x8
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldr x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br x11
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}

; CHECK-LABEL: .section .gfids$y,"dr"
; CHECK-NEXT:  .symidx escaped
; CHECK-NEXT:  .symidx $iexit_thunk$cdecl$v$v
; CHECK-NOT:   .symidx

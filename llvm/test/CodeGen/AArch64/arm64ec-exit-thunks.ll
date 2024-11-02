; RUN: llc -mtriple=arm64ec-pc-windows-msvc < %s | FileCheck %s

declare void @no_op() nounwind;
; CHECK-LABEL:    .def    $iexit_thunk$cdecl$v$v;
; CHECK:          .section        .wowthk$aa,"xr",discard,$iexit_thunk$cdecl$v$v
; CHECK:          // %bb.0:
; CHECK-NEXT:     sub     sp, sp, #48
; CHECK-NEXT:     .seh_stackalloc 48
; CHECK-NEXT:     stp     x29, x30, [sp, #32]             // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  32
; CHECK-NEXT:     add     x29, sp, #32
; CHECK-NEXT:     .seh_add_fp     32
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_call_no_redirect
; CHECK-NEXT:     ldr     x16, [x8, :lo12:__os_arm64x_dispatch_call_no_redirect]
; CHECK-NEXT:     blr     x16
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #32]             // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  32
; CHECK-NEXT:     add     sp, sp, #48
; CHECK-NEXT:     .seh_stackalloc 48
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     ret
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
; CHECK-LABEL:    .def    "#no_op$exit_thunk";
; CHECK:          .section        .wowthk$aa,"xr",discard,"#no_op$exit_thunk"
; CHECK:          .weak_anti_dep  no_op
; CHECK:          .weak_anti_dep  "#no_op"
; CHECK:          // %bb.0:
; CHECK-NEXT:     str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x8, __os_arm64x_check_icall
; CHECK-NEXT:     adrp    x11, no_op
; CHECK-NEXT:     add     x11, x11, :lo12:no_op
; CHECK-NEXT:     ldr     x8, [x8, :lo12:__os_arm64x_check_icall]
; CHECK-NEXT:     adrp    x10, ($iexit_thunk$cdecl$v$v)
; CHECK-NEXT:     add     x10, x10, :lo12:($iexit_thunk$cdecl$v$v)
; CHECK-NEXT:     blr     x8
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x11
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc

declare i64 @simple_integers(i8, i16, i32, i64) nounwind;
; CHECK-LABEL:    .def    $iexit_thunk$cdecl$i8$i8i8i8i8;
; CHECK:          .section        .wowthk$aa,"xr",discard,$iexit_thunk$cdecl$i8$i8i8i8i8
; CHECK:          // %bb.0:
; CHECK-NEXT:     sub     sp, sp, #48
; CHECK-NEXT:     .seh_stackalloc 48
; CHECK-NEXT:     stp     x29, x30, [sp, #32]             // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  32
; CHECK-NEXT:     add     x29, sp, #32
; CHECK-NEXT:     .seh_add_fp     32
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_call_no_redirect
; CHECK-NEXT:     ldr     x16, [x8, :lo12:__os_arm64x_dispatch_call_no_redirect]
; CHECK-NEXT:     blr     x16
; CHECK-NEXT:     mov     x0, x8
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #32]             // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  32
; CHECK-NEXT:     add     sp, sp, #48
; CHECK-NEXT:     .seh_stackalloc 48
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     ret
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
; CHECK-LABEL:    .def    "#simple_integers$exit_thunk";
; CHECK:          .section        .wowthk$aa,"xr",discard,"#simple_integers$exit_thunk"
; CHECK:          .weak_anti_dep  simple_integers
; CHECK:          .weak_anti_dep  "#simple_integers"
; CHECK:          // %bb.0:
; CHECK-NEXT:     str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x8, __os_arm64x_check_icall
; CHECK-NEXT:     adrp    x11, simple_integers
; CHECK-NEXT:     add     x11, x11, :lo12:simple_integers
; CHECK-NEXT:     ldr     x8, [x8, :lo12:__os_arm64x_check_icall]
; CHECK-NEXT:     adrp    x10, ($iexit_thunk$cdecl$i8$i8i8i8i8)
; CHECK-NEXT:     add     x10, x10, :lo12:($iexit_thunk$cdecl$i8$i8i8i8i8)
; CHECK-NEXT:     blr     x8
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x11
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc

; NOTE: Only float and double are supported.
declare double @simple_floats(float, double) nounwind;
; CHECK-LABEL:    .def    $iexit_thunk$cdecl$d$fd;
; CHECK:          .section        .wowthk$aa,"xr",discard,$iexit_thunk$cdecl$d$fd
; CHECK:          // %bb.0:
; CHECK-NEXT:     sub     sp, sp, #48
; CHECK-NEXT:     .seh_stackalloc 48
; CHECK-NEXT:     stp     x29, x30, [sp, #32]             // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  32
; CHECK-NEXT:     add     x29, sp, #32
; CHECK-NEXT:     .seh_add_fp     32
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_call_no_redirect
; CHECK-NEXT:     ldr     x16, [x8, :lo12:__os_arm64x_dispatch_call_no_redirect]
; CHECK-NEXT:     blr     x16
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #32]             // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  32
; CHECK-NEXT:     add     sp, sp, #48
; CHECK-NEXT:     .seh_stackalloc 48
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     ret
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
; CHECK-LABEL:    .def    "#simple_floats$exit_thunk";
; CHECK:          .section        .wowthk$aa,"xr",discard,"#simple_floats$exit_thunk"
; CHECK:          .weak_anti_dep  simple_floats
; CHECK:          .weak_anti_dep  "#simple_floats"
; CHECK:          // %bb.0:
; CHECK-NEXT:     str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x8, __os_arm64x_check_icall
; CHECK-NEXT:     adrp    x11, simple_floats
; CHECK-NEXT:     add     x11, x11, :lo12:simple_floats
; CHECK-NEXT:     ldr     x8, [x8, :lo12:__os_arm64x_check_icall]
; CHECK-NEXT:     adrp    x10, ($iexit_thunk$cdecl$d$fd)
; CHECK-NEXT:     add     x10, x10, :lo12:($iexit_thunk$cdecl$d$fd)
; CHECK-NEXT:     blr     x8
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x11
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc

declare void @has_varargs(...) nounwind;
; CHECK-LABEL:    .def    $iexit_thunk$cdecl$v$varargs;
; CHECK:          .section        .wowthk$aa,"xr",discard,$iexit_thunk$cdecl$v$varargs
; CHECK:          // %bb.0:
; CHECK-NEXT:     sub     sp, sp, #64
; CHECK-NEXT:     .seh_stackalloc 64
; CHECK-NEXT:     stp     x29, x30, [sp, #48]             // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  48
; CHECK-NEXT:     add     x29, sp, #48
; CHECK-NEXT:     .seh_add_fp     48
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_call_no_redirect
; CHECK-NEXT:     stp     x4, x5, [sp, #32]
; CHECK-NEXT:     ldr     x16, [x8, :lo12:__os_arm64x_dispatch_call_no_redirect]
; CHECK-NEXT:     blr     x16
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #48]             // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  48
; CHECK-NEXT:     add     sp, sp, #64
; CHECK-NEXT:     .seh_stackalloc 64
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     ret
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
; CHECK-LABEL:    .def    "#has_varargs$exit_thunk";
; CHECK:          .section        .wowthk$aa,"xr",discard,"#has_varargs$exit_thunk"
; CHECK:          .weak_anti_dep  has_varargs
; CHECK:          .weak_anti_dep  "#has_varargs"
; CHECK:          // %bb.0:
; CHECK-NEXT:     str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x8, __os_arm64x_check_icall
; CHECK-NEXT:     adrp    x11, has_varargs
; CHECK-NEXT:     add     x11, x11, :lo12:has_varargs
; CHECK-NEXT:     ldr     x8, [x8, :lo12:__os_arm64x_check_icall]
; CHECK-NEXT:     adrp    x10, ($iexit_thunk$cdecl$v$varargs)
; CHECK-NEXT:     add     x10, x10, :lo12:($iexit_thunk$cdecl$v$varargs)
; CHECK-NEXT:     blr     x8
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x11
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc

declare void @has_sret(ptr sret([100 x i8])) nounwind;
; CHECK-LABEL:    .def    $iexit_thunk$cdecl$m100$v;
; CHECK:          .section        .wowthk$aa,"xr",discard,$iexit_thunk$cdecl$m100$v
; CHECK:          // %bb.0:
; CHECK-NEXT:     sub     sp, sp, #48
; CHECK-NEXT:     .seh_stackalloc 48
; CHECK-NEXT:     stp     x29, x30, [sp, #32]             // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  32
; CHECK-NEXT:     add     x29, sp, #32
; CHECK-NEXT:     .seh_add_fp     32
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     mov     x0, x8
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_call_no_redirect
; CHECK-NEXT:     ldr     x16, [x8, :lo12:__os_arm64x_dispatch_call_no_redirect]
; CHECK-NEXT:     blr     x16
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #32]             // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  32
; CHECK-NEXT:     add     sp, sp, #48
; CHECK-NEXT:     .seh_stackalloc 48
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     ret
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
; CHECK-LABEL:    .def    "#has_sret$exit_thunk";
; CHECK:          .section        .wowthk$aa,"xr",discard,"#has_sret$exit_thunk"
; CHECK:          .weak_anti_dep  has_sret
; CHECK:          .weak_anti_dep  "#has_sret"
; CHECK:          // %bb.0:
; CHECK-NEXT:     str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x9, __os_arm64x_check_icall
; CHECK-NEXT:     adrp    x11, has_sret
; CHECK-NEXT:     add     x11, x11, :lo12:has_sret
; CHECK-NEXT:     ldr     x9, [x9, :lo12:__os_arm64x_check_icall]
; CHECK-NEXT:     adrp    x10, ($iexit_thunk$cdecl$m100$v)
; CHECK-NEXT:     add     x10, x10, :lo12:($iexit_thunk$cdecl$m100$v)
; CHECK-NEXT:     blr     x9
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x11
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc

%TSRet = type { i64, i64 }
declare void @has_aligned_sret(ptr align 32 sret(%TSRet)) nounwind;
; CHECK-LABEL:    .def    $iexit_thunk$cdecl$m16a32$v;
; CHECK:          .section        .wowthk$aa,"xr",discard,$iexit_thunk$cdecl$m16a32$v
; CHECK:          // %bb.0:
; CHECK-NEXT:     sub     sp, sp, #48
; CHECK-NEXT:     .seh_stackalloc 48
; CHECK-NEXT:     stp     x29, x30, [sp, #32]             // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  32
; CHECK-NEXT:     add     x29, sp, #32
; CHECK-NEXT:     .seh_add_fp     32
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     mov     x0, x8
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_call_no_redirect
; CHECK-NEXT:     ldr     x16, [x8, :lo12:__os_arm64x_dispatch_call_no_redirect]
; CHECK-NEXT:     blr     x16
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #32]             // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  32
; CHECK-NEXT:     add     sp, sp, #48
; CHECK-NEXT:     .seh_stackalloc 48
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     ret
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
; CHECK-LABEL:    .def    "#has_aligned_sret$exit_thunk";
; CHECK:          .section        .wowthk$aa,"xr",discard,"#has_aligned_sret$exit_thunk"
; CHECK:          .weak_anti_dep  has_aligned_sret
; CHECK:          .weak_anti_dep  "#has_aligned_sret"
; CHECK:          // %bb.0:
; CHECK:          str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK:          .seh_save_reg_x x30, 16
; CHECK:          .seh_endprologue
; CHECK:          adrp    x9, __os_arm64x_check_icall
; CHECK:          adrp    x11, has_aligned_sret
; CHECK:          add     x11, x11, :lo12:has_aligned_sret
; CHECK:          ldr     x9, [x9, :lo12:__os_arm64x_check_icall]
; CHECK:          adrp    x10, ($iexit_thunk$cdecl$m16a32$v)
; CHECK:          add     x10, x10, :lo12:($iexit_thunk$cdecl$m16a32$v)
; CHECK:          blr     x9
; CHECK:          .seh_startepilogue
; CHECK:          ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK:          .seh_save_reg_x x30, 16
; CHECK:          .seh_endepilogue
; CHECK:          br      x11
; CHECK:          .seh_endfunclet
; CHECK:          .seh_endproc

declare [2 x i8] @small_array([2 x i8], [2 x float]) nounwind;
; CHECK-LABEL:    .def    $iexit_thunk$cdecl$m2$m2F8;
; CHECK:          .section        .wowthk$aa,"xr",discard,$iexit_thunk$cdecl$m2$m2F8
; CHECK:          // %bb.0:
; CHECK-NEXT:     sub     sp, sp, #64
; CHECK-NEXT:     .seh_stackalloc 64
; CHECK-NEXT:     stp     x29, x30, [sp, #48]             // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  48
; CHECK-NEXT:     add     x29, sp, #48
; CHECK-NEXT:     .seh_add_fp     48
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     sturb   w1, [x29, #-1]
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_call_no_redirect
; CHECK-NEXT:     sturb   w0, [x29, #-2]
; CHECK-NEXT:     ldr     x16, [x8, :lo12:__os_arm64x_dispatch_call_no_redirect]
; CHECK-NEXT:     stp     s0, s1, [x29, #-12]
; CHECK-NEXT:     ldurh   w0, [x29, #-2]
; CHECK-NEXT:     ldur    x1, [x29, #-12]
; CHECK-NEXT:     blr     x16
; CHECK-NEXT:     mov     w0, w8
; CHECK-NEXT:     sturh   w8, [x29, #-14]
; CHECK-NEXT:     ubfx    w1, w8, #8, #8
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #48]             // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  48
; CHECK-NEXT:     add     sp, sp, #64
; CHECK-NEXT:     .seh_stackalloc 64
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     ret
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
; CHECK-LABEL:    .def    "#small_array$exit_thunk";
; CHECK:          .section        .wowthk$aa,"xr",discard,"#small_array$exit_thunk"
; CHECK:          .weak_anti_dep  small_array
; CHECK:          .weak_anti_dep  "#small_array"
; CHECK:          // %bb.0:
; CHECK-NEXT:     str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x8, __os_arm64x_check_icall
; CHECK-NEXT:     adrp    x11, small_array
; CHECK-NEXT:     add     x11, x11, :lo12:small_array
; CHECK-NEXT:     ldr     x8, [x8, :lo12:__os_arm64x_check_icall]
; CHECK-NEXT:     adrp    x10, ($iexit_thunk$cdecl$m2$m2F8)
; CHECK-NEXT:     add     x10, x10, :lo12:($iexit_thunk$cdecl$m2$m2F8)
; CHECK-NEXT:     blr     x8
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x11
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc

declare [3 x i64] @large_array([3 x i64], [2 x double], [2 x [2 x i64]]) nounwind;
; CHECK-LABEL:    .def    $iexit_thunk$cdecl$m24$m24D16m32;
; CHECK:          .section        .wowthk$aa,"xr",discard,$iexit_thunk$cdecl$m24$m24D16m32
; CHECK:          // %bb.0:
; CHECK-NEXT:     sub     sp, sp, #144
; CHECK-NEXT:     .seh_stackalloc 144
; CHECK-NEXT:     stp     x29, x30, [sp, #128]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  128
; CHECK-NEXT:     add     x29, sp, #128
; CHECK-NEXT:     .seh_add_fp     128
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_call_no_redirect
; CHECK-NEXT:     stp     x0, x1, [x29, #-48]
; CHECK-NEXT:     sub     x0, x29, #24
; CHECK-NEXT:     ldr     x16, [x8, :lo12:__os_arm64x_dispatch_call_no_redirect]
; CHECK-NEXT:     stur    x2, [x29, #-32]
; CHECK-NEXT:     sub     x1, x29, #48
; CHECK-NEXT:     stp     x3, x4, [sp, #32]
; CHECK-NEXT:     add     x2, sp, #64
; CHECK-NEXT:     add     x3, sp, #32
; CHECK-NEXT:     stp     d0, d1, [sp, #64]
; CHECK-NEXT:     stp     x5, x6, [sp, #48]
; CHECK-NEXT:     blr     x16
; CHECK-NEXT:     ldp     x0, x1, [x29, #-24]
; CHECK-NEXT:     ldur    x2, [x29, #-8]
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #128]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  128
; CHECK-NEXT:     add     sp, sp, #144
; CHECK-NEXT:     .seh_stackalloc 144
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     ret
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
; CHECK-LABEL:    .def    "#large_array$exit_thunk";
; CHECK:          .section        .wowthk$aa,"xr",discard,"#large_array$exit_thunk"
; CHECK:          .weak_anti_dep  large_array
; CHECK:          .weak_anti_dep  "#large_array"
; CHECK:          // %bb.0:
; CHECK-NEXT:     str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x8, __os_arm64x_check_icall
; CHECK-NEXT:     adrp    x11, large_array
; CHECK-NEXT:     add     x11, x11, :lo12:large_array
; CHECK-NEXT:     ldr     x8, [x8, :lo12:__os_arm64x_check_icall]
; CHECK-NEXT:     adrp    x10, ($iexit_thunk$cdecl$m24$m24D16m32)
; CHECK-NEXT:     add     x10, x10, :lo12:($iexit_thunk$cdecl$m24$m24D16m32)
; CHECK-NEXT:     blr     x8
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x11
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc

%T1 = type { i16 }
%T2 = type { i32, float }
%T3 = type { i64, double }
%T4 = type { i64, double, i8 }
declare %T2 @simple_struct(%T1, %T2, %T3, %T4) nounwind;
; CHECK-LABEL:    .def    $iexit_thunk$cdecl$m8$i8m8m16m24;
; CHECK:          .section        .wowthk$aa,"xr",discard,$iexit_thunk$cdecl$m8$i8m8m16m24
; CHECK:          // %bb.0:
; CHECK-NEXT:     sub     sp, sp, #112
; CHECK-NEXT:     .seh_stackalloc 112
; CHECK-NEXT:     stp     x29, x30, [sp, #96]             // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  96
; CHECK-NEXT:     add     x29, sp, #96
; CHECK-NEXT:     .seh_add_fp     96
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     stur    w1, [x29, #-8]
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_call_no_redirect
; CHECK-NEXT:     stur    s0, [x29, #-4]
; CHECK-NEXT:     ldr     x16, [x8, :lo12:__os_arm64x_dispatch_call_no_redirect]
; CHECK-NEXT:     ldur    x1, [x29, #-8]
; CHECK-NEXT:     stur    x2, [x29, #-24]
; CHECK-NEXT:     sub     x2, x29, #24
; CHECK-NEXT:     str     x3, [sp, #48]
; CHECK-NEXT:     add     x3, sp, #48
; CHECK-NEXT:     stur    d1, [x29, #-16]
; CHECK-NEXT:     str     d2, [sp, #56]
; CHECK-NEXT:     strb    w4, [sp, #64]
; CHECK-NEXT:     blr     x16
; CHECK-NEXT:     str     x8, [sp, #40]
; CHECK-NEXT:     mov     x0, x8
; CHECK-NEXT:     ldr     s0, [sp, #44]
; CHECK-NEXT:                                     // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #96]             // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  96
; CHECK-NEXT:     add     sp, sp, #112
; CHECK-NEXT:     .seh_stackalloc 112
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     ret
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
; CHECK-LABEL:    .def    "#simple_struct$exit_thunk";
; CHECK:          .section        .wowthk$aa,"xr",discard,"#simple_struct$exit_thunk"
; CHECK:          .weak_anti_dep  simple_struct
; CHECK:          .weak_anti_dep  "#simple_struct"
; CHECK:          // %bb.0:
; CHECK-NEXT:     str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     adrp    x8, __os_arm64x_check_icall
; CHECK-NEXT:     adrp    x11, simple_struct
; CHECK-NEXT:     add     x11, x11, :lo12:simple_struct
; CHECK-NEXT:     ldr     x8, [x8, :lo12:__os_arm64x_check_icall]
; CHECK-NEXT:     adrp    x10, ($iexit_thunk$cdecl$m8$i8m8m16m24)
; CHECK-NEXT:     add     x10, x10, :lo12:($iexit_thunk$cdecl$m8$i8m8m16m24)
; CHECK-NEXT:     blr     x8
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:     .seh_save_reg_x x30, 16
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x11
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc

; CHECK-LABEL:    .section        .hybmp$x,"yi"
; CHECK-NEXT:     .symidx "#func_caller"
; CHECK-NEXT:     .symidx $ientry_thunk$cdecl$v$v
; CHECK-NEXT:     .word   1
; CHECK-NEXT:     .symidx no_op
; CHECK-NEXT:     .symidx $iexit_thunk$cdecl$v$v
; CHECK-NEXT:     .word   4
; CHECK-NEXT:     .symidx "#no_op$exit_thunk"
; CHECK-NEXT:     .symidx no_op
; CHECK-NEXT:     .word   0
; CHECK-NEXT:     .symidx simple_integers
; CHECK-NEXT:     .symidx $iexit_thunk$cdecl$i8$i8i8i8i8
; CHECK-NEXT:     .word   4
; CHECK-NEXT:     .symidx "#simple_integers$exit_thunk"
; CHECK-NEXT:     .symidx simple_integers
; CHECK-NEXT:     .word   0
; CHECK-NEXT:     .symidx simple_floats
; CHECK-NEXT:     .symidx $iexit_thunk$cdecl$d$fd
; CHECK-NEXT:     .word   4
; CHECK-NEXT:     .symidx "#simple_floats$exit_thunk"
; CHECK-NEXT:     .symidx simple_floats
; CHECK-NEXT:     .word   0
; CHECK-NEXT:     .symidx has_varargs
; CHECK-NEXT:     .symidx $iexit_thunk$cdecl$v$varargs
; CHECK-NEXT:     .word   4
; CHECK-NEXT:     .symidx "#has_varargs$exit_thunk"
; CHECK-NEXT:     .symidx has_varargs
; CHECK-NEXT:     .word 0
; CHECK-NEXT:     .symidx has_sret
; CHECK-NEXT:     .symidx $iexit_thunk$cdecl$m100$v
; CHECK-NEXT:     .word   4
; CHECK-NEXT:     .symidx "#has_sret$exit_thunk"
; CHECK-NEXT:     .symidx has_sret
; CHECK-NEXT:     .word   0
; CHECK-NEXT:     .symidx has_aligned_sret
; CHECK-NEXT:     .symidx $iexit_thunk$cdecl$m16a32$v
; CHECK-NEXT:     .word   4
; CHECK-NEXT:     .symidx "#has_aligned_sret$exit_thunk"
; CHECK-NEXT:     .symidx has_aligned_sret
; CHECK-NEXT:     .word   0
; CHECK-NEXT:     .symidx small_array
; CHECK-NEXT:     .symidx $iexit_thunk$cdecl$m2$m2F8
; CHECK-NEXT:     .word   4
; CHECK-NEXT:     .symidx "#small_array$exit_thunk"
; CHECK-NEXT:     .symidx small_array
; CHECK-NEXT:     .word   0
; CHECK-NEXT:     .symidx large_array
; CHECK-NEXT:     .symidx $iexit_thunk$cdecl$m24$m24D16m32
; CHECK-NEXT:     .word   4
; CHECK-NEXT:     .symidx "#large_array$exit_thunk"
; CHECK-NEXT:     .symidx large_array
; CHECK-NEXT:     .word   0
; CHECK-NEXT:     .symidx simple_struct
; CHECK-NEXT:     .symidx $iexit_thunk$cdecl$m8$i8m8m16m24
; CHECK-NEXT:     .word   4
; CHECK-NEXT:     .symidx "#simple_struct$exit_thunk"
; CHECK-NEXT:     .symidx simple_struct
; CHECK-NEXT:     .word   0

define void @func_caller() nounwind {
  call void @no_op()
  call i64 @simple_integers(i8 0, i16 0, i32 0, i64 0)
  call double @simple_floats(float 0.0, double 0.0)
  call void (...) @has_varargs()
  %c = alloca i8
  call void @has_sret(ptr sret([100 x i8]) %c)
  %aligned = alloca %TSRet, align 32
  store %TSRet { i64 0, i64 0 }, ptr %aligned, align 32
  call void @has_aligned_sret(ptr align 32 sret(%TSRet) %aligned)
  call [2 x i8] @small_array([2 x i8] [i8 0, i8 0], [2 x float] [float 0.0, float 0.0])
  call [3 x i64] @large_array([3 x i64] [i64 0, i64 0, i64 0], [2 x double] [double 0.0, double 0.0], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 0], [2 x i64] [i64 0, i64 0]])
  call %T2 @simple_struct(%T1 { i16 0 }, %T2 { i32 0, float 0.0 }, %T3 { i64 0, double 0.0 }, %T4 { i64 0, double 0.0, i8 0 })
  ret void
}

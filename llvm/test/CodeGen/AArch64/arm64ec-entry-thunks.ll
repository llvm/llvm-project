; RUN: llc -mtriple=arm64ec-pc-windows-msvc < %s | FileCheck %s

define void @no_op() nounwind {
; CHECK-LABEL     .def    $ientry_thunk$cdecl$v$v;
; CHECK:          .section        .wowthk$aa,"xr",discard,$ientry_thunk$cdecl$v$v
; CHECK:          // %bb.0:
; CHECK-NEXT:     stp     q6, q7, [sp, #-176]!            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 176
; CHECK-NEXT:     stp     q8, q9, [sp, #32]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     stp     q10, q11, [sp, #64]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     stp     q12, q13, [sp, #96]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     stp     q14, q15, [sp, #128]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     stp     x29, x30, [sp, #160]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  160
; CHECK-NEXT:     add     x29, sp, #160
; CHECK-NEXT:     .seh_add_fp     160
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     blr     x9
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_ret
; CHECK-NEXT:     ldr     x0, [x8, :lo12:__os_arm64x_dispatch_ret]
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #160]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  160
; CHECK-NEXT:     ldp     q14, q15, [sp, #128]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     ldp     q12, q13, [sp, #96]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     ldp     q10, q11, [sp, #64]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     ldp     q8, q9, [sp, #32]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     ldp     q6, q7, [sp], #176              // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 176
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x0
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
  ret void
}

define i64 @simple_integers(i8, i16, i32, i64) nounwind {
; CHECK-LABEL:    .def    $ientry_thunk$cdecl$i8$i8i8i8i8;
; CHECK:          .section        .wowthk$aa,"xr",discard,$ientry_thunk$cdecl$i8$i8i8i8i8
; CHECK:          // %bb.0:
; CHECK-NEXT:     stp     q6, q7, [sp, #-176]!            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 176
; CHECK-NEXT:     stp     q8, q9, [sp, #32]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     stp     q10, q11, [sp, #64]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     stp     q12, q13, [sp, #96]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     stp     q14, q15, [sp, #128]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     stp     x29, x30, [sp, #160]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  160
; CHECK-NEXT:     add     x29, sp, #160
; CHECK-NEXT:     .seh_add_fp     160
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     blr     x9
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_ret
; CHECK-NEXT:     ldr     x1, [x8, :lo12:__os_arm64x_dispatch_ret]
; CHECK-NEXT:     mov     x8, x0
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #160]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  160
; CHECK-NEXT:     ldp     q14, q15, [sp, #128]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     ldp     q12, q13, [sp, #96]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     ldp     q10, q11, [sp, #64]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     ldp     q8, q9, [sp, #32]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     ldp     q6, q7, [sp], #176              // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 176
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x1
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
  ret i64 0
}

; NOTE: Only float and double are supported.
define double @simple_floats(float, double) nounwind {
; CHECK-LABEL:    .def    $ientry_thunk$cdecl$d$fd;
; CHECK:          .section        .wowthk$aa,"xr",discard,$ientry_thunk$cdecl$d$fd
; CHECK:          // %bb.0:
; CHECK-NEXT:     stp     q6, q7, [sp, #-176]!            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 176
; CHECK-NEXT:     stp     q8, q9, [sp, #32]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     stp     q10, q11, [sp, #64]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     stp     q12, q13, [sp, #96]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     stp     q14, q15, [sp, #128]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     stp     x29, x30, [sp, #160]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  160
; CHECK-NEXT:     add     x29, sp, #160
; CHECK-NEXT:     .seh_add_fp     160
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     blr     x9
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_ret
; CHECK-NEXT:     ldr     x0, [x8, :lo12:__os_arm64x_dispatch_ret]
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #160]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  160
; CHECK-NEXT:     ldp     q14, q15, [sp, #128]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     ldp     q12, q13, [sp, #96]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     ldp     q10, q11, [sp, #64]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     ldp     q8, q9, [sp, #32]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     ldp     q6, q7, [sp], #176              // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 176
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x0
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
  ret double 0.0
}

define void @has_varargs(...) nounwind {
; CHECK-LABEL:    .def    $ientry_thunk$cdecl$v$varargs;
; CHECK:          .section        .wowthk$aa,"xr",discard,$ientry_thunk$cdecl$v$varargs
; CHECK:          // %bb.0:
; CHECK-NEXT:     stp     q6, q7, [sp, #-176]!            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 176
; CHECK-NEXT:     stp     q8, q9, [sp, #32]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     stp     q10, q11, [sp, #64]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     stp     q12, q13, [sp, #96]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     stp     q14, q15, [sp, #128]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     stp     x29, x30, [sp, #160]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  160
; CHECK-NEXT:     add     x29, sp, #160
; CHECK-NEXT:     .seh_add_fp     160
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     add     x4, x4, #32
; CHECK-NEXT:     mov     x5, xzr
; CHECK-NEXT:     blr     x9
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_ret
; CHECK-NEXT:     ldr     x0, [x8, :lo12:__os_arm64x_dispatch_ret]
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #160]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  160
; CHECK-NEXT:     ldp     q14, q15, [sp, #128]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     ldp     q12, q13, [sp, #96]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     ldp     q10, q11, [sp, #64]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     ldp     q8, q9, [sp, #32]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     ldp     q6, q7, [sp], #176              // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 176
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x0
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
  ret void
}

define void @has_sret(ptr sret([100 x i8])) nounwind {
; CHECK-LABEL:    .def    $ientry_thunk$cdecl$i8$v;
; CHECK:          .section        .wowthk$aa,"xr",discard,$ientry_thunk$cdecl$i8$v
; CHECK:          // %bb.0:
; CHECK-NEXT:     stp     q6, q7, [sp, #-176]!            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 176
; CHECK-NEXT:     stp     q8, q9, [sp, #32]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     stp     q10, q11, [sp, #64]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     stp     q12, q13, [sp, #96]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     stp     q14, q15, [sp, #128]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     stp     x29, x30, [sp, #160]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  160
; CHECK-NEXT:     add     x29, sp, #160
; CHECK-NEXT:     .seh_add_fp     160
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     blr     x9
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_ret
; CHECK-NEXT:     ldr     x1, [x8, :lo12:__os_arm64x_dispatch_ret]
; CHECK-NEXT:     mov     x8, x0
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #160]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  160
; CHECK-NEXT:     ldp     q14, q15, [sp, #128]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     ldp     q12, q13, [sp, #96]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     ldp     q10, q11, [sp, #64]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     ldp     q8, q9, [sp, #32]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     ldp     q6, q7, [sp], #176              // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 176
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x1
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
  ret void
}

define i8 @matches_has_sret() nounwind {
; Verify that $ientry_thunk$cdecl$i8$v is re-used by a function with matching signature.
; CHECK-NOT: .def    $ientry_thunk$cdecl$i8$v;
  ret i8 0
}

%TSRet = type { i64, i64 }
define void @has_aligned_sret(ptr align 32 sret(%TSRet), i32) nounwind {
; CHECK-LABEL:    .def    $ientry_thunk$cdecl$m16$i8;
; CHECK:          .section        .wowthk$aa,"xr",discard,$ientry_thunk$cdecl$m16$i8
; CHECK:          // %bb.0:
; CHECK-NEXT:     stp     q6, q7, [sp, #-192]!            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 192
; CHECK-NEXT:     stp     q8, q9, [sp, #32]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     stp     q10, q11, [sp, #64]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     stp     q12, q13, [sp, #96]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     stp     q14, q15, [sp, #128]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     str     x19, [sp, #160]                 // 8-byte Folded Spill
; CHECK-NEXT:     .seh_save_reg   x19, 160
; CHECK-NEXT:     stp     x29, x30, [sp, #168]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  168
; CHECK-NEXT:     add     x29, sp, #168
; CHECK-NEXT:     .seh_add_fp     168
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     mov     x19, x0
; CHECK-NEXT:     mov     x8, x0
; CHECK-NEXT:     mov     x0, x1
; CHECK-NEXT:     blr     x9
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_ret
; CHECK-NEXT:     ldr     x0, [x8, :lo12:__os_arm64x_dispatch_ret]
; CHECK-NEXT:     mov     x8, x19
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #168]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  168
; CHECK-NEXT:     ldr     x19, [sp, #160]                 // 8-byte Folded Reload
; CHECK-NEXT:     .seh_save_reg   x19, 160
; CHECK-NEXT:     ldp     q14, q15, [sp, #128]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     ldp     q12, q13, [sp, #96]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     ldp     q10, q11, [sp, #64]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     ldp     q8, q9, [sp, #32]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     ldp     q6, q7, [sp], #192              // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 192
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x0
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
  ret void
}

define [2 x i8] @small_array([2 x i8] %0, [2 x float]) nounwind {
; CHECK-LABEL:    .def    $ientry_thunk$cdecl$m2$m2F8;
; CHECK:          .section        .wowthk$aa,"xr",discard,$ientry_thunk$cdecl$m2$m2F8
; CHECK:          // %bb.0:
; CHECK-NEXT:     sub     sp, sp, #192
; CHECK-NEXT:     .seh_stackalloc 192
; CHECK-NEXT:     stp     q6, q7, [sp, #16]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q6, 16
; CHECK-NEXT:     stp     q8, q9, [sp, #48]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 48
; CHECK-NEXT:     stp     q10, q11, [sp, #80]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 80
; CHECK-NEXT:     stp     q12, q13, [sp, #112]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 112
; CHECK-NEXT:     stp     q14, q15, [sp, #144]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 144
; CHECK-NEXT:     stp     x29, x30, [sp, #176]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  176
; CHECK-NEXT:     add     x29, sp, #176
; CHECK-NEXT:     .seh_add_fp     176
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     stur    x1, [sp, #4]
; CHECK-NEXT:     ubfx    w1, w0, #8, #8
; CHECK-NEXT:     ldp     s0, s1, [sp, #4]
; CHECK-NEXT:     strh    w0, [sp, #14]
; CHECK-NEXT:     blr     x9
; CHECK-NEXT:     adrp    x9, __os_arm64x_dispatch_ret
; CHECK-NEXT:     strb    w0, [sp, #2]
; CHECK-NEXT:     strb    w1, [sp, #3]
; CHECK-NEXT:     ldrh    w8, [sp, #2]
; CHECK-NEXT:     ldr     x0, [x9, :lo12:__os_arm64x_dispatch_ret]
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #176]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  176
; CHECK-NEXT:     ldp     q14, q15, [sp, #144]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 144
; CHECK-NEXT:     ldp     q12, q13, [sp, #112]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 112
; CHECK-NEXT:     ldp     q10, q11, [sp, #80]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 80
; CHECK-NEXT:     ldp     q8, q9, [sp, #48]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 48
; CHECK-NEXT:     ldp     q6, q7, [sp, #16]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q6, 16
; CHECK-NEXT:     add     sp, sp, #192
; CHECK-NEXT:     .seh_stackalloc 192
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x0
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
  ret [2 x i8] %0
}

define [3 x i64] @large_array([3 x i64] %0, [2 x double], [2 x [2 x i64]]) nounwind {
; CHECK-LABEL:    .def    $ientry_thunk$cdecl$m24$m24D16m32;
; CHECK:          .section        .wowthk$aa,"xr",discard,$ientry_thunk$cdecl$m24$m24D16m32
; CHECK:          // %bb.0:
; CHECK-NEXT:     stp     q6, q7, [sp, #-192]!            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 192
; CHECK-NEXT:     stp     q8, q9, [sp, #32]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     stp     q10, q11, [sp, #64]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     stp     q12, q13, [sp, #96]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     stp     q14, q15, [sp, #128]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     str     x19, [sp, #160]                 // 8-byte Folded Spill
; CHECK-NEXT:     .seh_save_reg   x19, 160
; CHECK-NEXT:     stp     x29, x30, [sp, #168]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  168
; CHECK-NEXT:     add     x29, sp, #168
; CHECK-NEXT:     .seh_add_fp     168
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     ldp     x10, x8, [x1, #8]
; CHECK-NEXT:     mov     x19, x0
; CHECK-NEXT:     ldp     d0, d1, [x2]
; CHECK-NEXT:     ldr     x0, [x1]
; CHECK-NEXT:     ldp     x5, x6, [x3, #16]
; CHECK-NEXT:     ldp     x3, x4, [x3]
; CHECK-NEXT:     mov     x1, x10
; CHECK-NEXT:     mov     x2, x8
; CHECK-NEXT:     blr     x9
; CHECK-NEXT:     stp     x0, x1, [x19]
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_ret
; CHECK-NEXT:     str     x2, [x19, #16]
; CHECK-NEXT:     ldr     x0, [x8, :lo12:__os_arm64x_dispatch_ret]
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #168]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  168
; CHECK-NEXT:     ldr     x19, [sp, #160]                 // 8-byte Folded Reload
; CHECK-NEXT:     .seh_save_reg   x19, 160
; CHECK-NEXT:     ldp     q14, q15, [sp, #128]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     ldp     q12, q13, [sp, #96]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     ldp     q10, q11, [sp, #64]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     ldp     q8, q9, [sp, #32]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     ldp     q6, q7, [sp], #192              // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 192
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x0
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
  ret [3 x i64] %0
}

%T1 = type { i16 }
%T2 = type { i32, float }
%T3 = type { i64, double }
%T4 = type { i64, double, i8 }
define %T2 @simple_struct(%T1 %0, %T2 %1, %T3, %T4) nounwind {
; CHECK-LABEL:    .def    $ientry_thunk$cdecl$m8$i8m8m16m24;
; CHECK:          .section        .wowthk$aa,"xr",discard,$ientry_thunk$cdecl$m8$i8m8m16m24
; CHECK:          // %bb.0:
; CHECK-NEXT:     sub     sp, sp, #192
; CHECK-NEXT:     .seh_stackalloc 192
; CHECK-NEXT:     stp     q6, q7, [sp, #16]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q6, 16
; CHECK-NEXT:     stp     q8, q9, [sp, #48]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 48
; CHECK-NEXT:     stp     q10, q11, [sp, #80]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 80
; CHECK-NEXT:     stp     q12, q13, [sp, #112]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 112
; CHECK-NEXT:     stp     q14, q15, [sp, #144]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 144
; CHECK-NEXT:     stp     x29, x30, [sp, #176]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  176
; CHECK-NEXT:     add     x29, sp, #176
; CHECK-NEXT:     .seh_add_fp     176
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     str     x1, [sp, #8]
; CHECK-NEXT:     ldr     x8, [x2]
; CHECK-NEXT:     ldr     x10, [x3]
; CHECK-NEXT:     ldr     d1, [x2, #8]
; CHECK-NEXT:                                     // kill: def $w1 killed $w1 killed $x1
; CHECK-NEXT:     ldr     s0, [sp, #12]
; CHECK-NEXT:     ldr     d2, [x3, #8]
; CHECK-NEXT:     mov     x2, x8
; CHECK-NEXT:     ldrb    w4, [x3, #16]
; CHECK-NEXT:     mov     x3, x10
; CHECK-NEXT:     blr     x9
; CHECK-NEXT:     adrp    x9, __os_arm64x_dispatch_ret
; CHECK-NEXT:     str     w0, [sp]
; CHECK-NEXT:     str     s0, [sp, #4]
; CHECK-NEXT:     ldr     x8, [sp]
; CHECK-NEXT:     ldr     x0, [x9, :lo12:__os_arm64x_dispatch_ret]
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #176]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  176
; CHECK-NEXT:     ldp     q14, q15, [sp, #144]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 144
; CHECK-NEXT:     ldp     q12, q13, [sp, #112]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 112
; CHECK-NEXT:     ldp     q10, q11, [sp, #80]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 80
; CHECK-NEXT:     ldp     q8, q9, [sp, #48]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 48
; CHECK-NEXT:     ldp     q6, q7, [sp, #16]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q6, 16
; CHECK-NEXT:     add     sp, sp, #192
; CHECK-NEXT:     .seh_stackalloc 192
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x0
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
  ret %T2 %1
}

define void @cxx_method(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr dead_on_unwind inreg noalias writable sret(i64) align 8 %1) {
; CHECK-LABEL:    .def    $ientry_thunk$cdecl$i8$i8i8;
; CHECK: .section        .wowthk$aa,"xr",discard,$ientry_thunk$cdecl$i8$i8i8
; CHECK:          // %bb.0:
; CHECK-NEXT:     stp     q6, q7, [sp, #-176]!            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 176
; CHECK-NEXT:     stp     q8, q9, [sp, #32]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     stp     q10, q11, [sp, #64]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     stp     q12, q13, [sp, #96]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     stp     q14, q15, [sp, #128]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     stp     x29, x30, [sp, #160]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr  160
; CHECK-NEXT:     add     x29, sp, #160
; CHECK-NEXT:     .seh_add_fp     160
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     blr     x9
; CHECK-NEXT:     adrp    x8, __os_arm64x_dispatch_ret
; CHECK-NEXT:     ldr     x1, [x8, :lo12:__os_arm64x_dispatch_ret]
; CHECK-NEXT:     mov     x8, x0
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp     x29, x30, [sp, #160]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr  160
; CHECK-NEXT:     ldp     q14, q15, [sp, #128]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q14, 128
; CHECK-NEXT:     ldp     q12, q13, [sp, #96]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q12, 96
; CHECK-NEXT:     ldp     q10, q11, [sp, #64]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q10, 64
; CHECK-NEXT:     ldp     q8, q9, [sp, #32]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p     q8, 32
; CHECK-NEXT:     ldp     q6, q7, [sp], #176              // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_px    q6, 176
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x1
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
  ret void
}

define <4 x i8> @small_vector(<4 x i8> %0) {
; CHECK-LABEL:    .def	$ientry_thunk$cdecl$m$m;
; CHECK:          .section	.wowthk$aa,"xr",discard,$ientry_thunk$cdecl$m$m
; CHECK:          // %bb.0:
; CHECK-NEXT:     sub	sp, sp, #192
; CHECK-NEXT:     .seh_stackalloc	192
; CHECK-NEXT:     stp	q6, q7, [sp, #16]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p	q6, 16
; CHECK-NEXT:     stp	q8, q9, [sp, #48]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p	q8, 48
; CHECK-NEXT:     stp	q10, q11, [sp, #80]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p	q10, 80
; CHECK-NEXT:     stp	q12, q13, [sp, #112]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p	q12, 112
; CHECK-NEXT:     stp	q14, q15, [sp, #144]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p	q14, 144
; CHECK-NEXT:     stp	x29, x30, [sp, #176]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr	176
; CHECK-NEXT:     add	x29, sp, #176
; CHECK-NEXT:     .seh_add_fp	176
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     str	w0, [sp, #12]
; CHECK-NEXT:     ldr	s0, [sp, #12]
; CHECK-NEXT:     ushll	v0.8h, v0.8b, #0
; CHECK-NEXT:                                           // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:     blr	x9
; CHECK-NEXT:     uzp1	v0.8b, v0.8b, v0.8b
; CHECK-NEXT:     adrp	x9, __os_arm64x_dispatch_ret
; CHECK-NEXT:     str	s0, [sp, #8]
; CHECK-NEXT:     fmov	w8, s0
; CHECK-NEXT:     ldr	x0, [x9, :lo12:__os_arm64x_dispatch_ret]
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp	x29, x30, [sp, #176]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr	176
; CHECK-NEXT:     ldp	q14, q15, [sp, #144]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p	q14, 144
; CHECK-NEXT:     ldp	q12, q13, [sp, #112]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p	q12, 112
; CHECK-NEXT:     ldp	q10, q11, [sp, #80]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p	q10, 80
; CHECK-NEXT:     ldp	q8, q9, [sp, #48]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p	q8, 48
; CHECK-NEXT:     ldp	q6, q7, [sp, #16]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p	q6, 16
; CHECK-NEXT:     add	sp, sp, #192
; CHECK-NEXT:     .seh_stackalloc	192
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br	x0
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
start:
  ret <4 x i8> %0
}

define <8 x i16> @large_vector(<8 x i16> %0) {
; CHECK-LABEL:    .def	$ientry_thunk$cdecl$m16$m16;
; CHECK:          .section	.wowthk$aa,"xr",discard,$ientry_thunk$cdecl$m16$m16
; CHECK:          // %bb.0:
; CHECK-NEXT:     stp	q6, q7, [sp, #-192]!            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_px	q6, 192
; CHECK-NEXT:     stp	q8, q9, [sp, #32]               // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p	q8, 32
; CHECK-NEXT:     stp	q10, q11, [sp, #64]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p	q10, 64
; CHECK-NEXT:     stp	q12, q13, [sp, #96]             // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p	q12, 96
; CHECK-NEXT:     stp	q14, q15, [sp, #128]            // 32-byte Folded Spill
; CHECK-NEXT:     .seh_save_any_reg_p	q14, 128
; CHECK-NEXT:     str	x19, [sp, #160]                 // 8-byte Folded Spill
; CHECK-NEXT:     .seh_save_reg	x19, 160
; CHECK-NEXT:     stp	x29, x30, [sp, #168]            // 16-byte Folded Spill
; CHECK-NEXT:     .seh_save_fplr	168
; CHECK-NEXT:     add	x29, sp, #168
; CHECK-NEXT:     .seh_add_fp	168
; CHECK-NEXT:     .seh_endprologue
; CHECK-NEXT:     ldr	q0, [x1]
; CHECK-NEXT:     mov	x19, x0
; CHECK-NEXT:     blr	x9
; CHECK-NEXT:     adrp	x8, __os_arm64x_dispatch_ret
; CHECK-NEXT:     str	q0, [x19]
; CHECK-NEXT:     ldr	x0, [x8, :lo12:__os_arm64x_dispatch_ret]
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldp	x29, x30, [sp, #168]            // 16-byte Folded Reload
; CHECK-NEXT:     .seh_save_fplr	168
; CHECK-NEXT:     ldr	x19, [sp, #160]                 // 8-byte Folded Reload
; CHECK-NEXT:     .seh_save_reg	x19, 160
; CHECK-NEXT:     ldp	q14, q15, [sp, #128]            // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p	q14, 128
; CHECK-NEXT:     ldp	q12, q13, [sp, #96]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p	q12, 96
; CHECK-NEXT:     ldp	q10, q11, [sp, #64]             // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p	q10, 64
; CHECK-NEXT:     ldp	q8, q9, [sp, #32]               // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_p	q8, 32
; CHECK-NEXT:     ldp	q6, q7, [sp], #192              // 32-byte Folded Reload
; CHECK-NEXT:     .seh_save_any_reg_px	q6, 192
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br	x0
; CHECK-NEXT:     .seh_endfunclet
; CHECK-NEXT:     .seh_endproc
start:
  ret <8 x i16> %0
}

; Verify the hybrid bitmap
; CHECK-LABEL:    .section        .hybmp$x,"yi"
; CHECK-NEXT:     .symidx "#no_op"
; CHECK-NEXT:     .symidx $ientry_thunk$cdecl$v$v
; CHECK-NEXT:     .word   1
; CHECK-NEXT:     .symidx "#simple_integers"
; CHECK-NEXT:     .symidx $ientry_thunk$cdecl$i8$i8i8i8i8
; CHECK-NEXT:     .word   1
; CHECK-NEXT:     .symidx "#simple_floats"
; CHECK-NEXT:     .symidx $ientry_thunk$cdecl$d$fd
; CHECK-NEXT:     .word   1
; CHECK-NEXT:     .symidx "#has_varargs"
; CHECK-NEXT:     .symidx $ientry_thunk$cdecl$v$varargs
; CHECK-NEXT:     .word   1
; CHECK-NEXT:     .symidx "#has_sret"
; CHECK-NEXT:     .symidx $ientry_thunk$cdecl$m100$v
; CHECK-NEXT:     .word   1
; CHECK-NEXT:     .symidx "#matches_has_sret"
; CHECK-NEXT:     .symidx $ientry_thunk$cdecl$i8$v
; CHECK-NEXT:     .word   1
; CHECK-NEXT:     .symidx "#has_aligned_sret"
; CHECK-NEXT:     .symidx $ientry_thunk$cdecl$m16$i8
; CHECK-NEXT:     .word   1
; CHECK-NEXT:     .symidx "#small_array"
; CHECK-NEXT:     .symidx $ientry_thunk$cdecl$m2$m2F8
; CHECK-NEXT:     .word   1
; CHECK-NEXT:     .symidx "#large_array"
; CHECK-NEXT:     .symidx $ientry_thunk$cdecl$m24$m24D16m32
; CHECK-NEXT:     .word   1
; CHECK-NEXT:     .symidx "#simple_struct"
; CHECK-NEXT:     .symidx $ientry_thunk$cdecl$m8$i8m8m16m24
; CHECK-NEXT:     .word   1
; CHECK-NEXT:     .symidx "#cxx_method"
; CHECK-NEXT:     .symidx $ientry_thunk$cdecl$i8$i8i8
; CHECK-NEXT:     .word   1
; CHECK-NEXT:     .symidx	"#small_vector"
; CHECK-NEXT:     .symidx	$ientry_thunk$cdecl$m$m
; CHECK-NEXT:     .word	1
; CHECK-NEXT:     .symidx	"#large_vector"
; CHECK-NEXT:     .symidx	$ientry_thunk$cdecl$m16$m16
; CHECK-NEXT:     .word	1

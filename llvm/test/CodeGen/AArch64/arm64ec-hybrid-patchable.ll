; RUN: llc -mtriple=arm64ec-pc-windows-msvc < %s | FileCheck %s
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -t %t.o | FileCheck --check-prefix=SYM %s

define dso_local ptr @func() hybrid_patchable nounwind {
; SYM: [ 8](sec  4)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #func$hp_target
; CHECK-LABEL:     .def    "#func$hp_target";
; CHECK:           .section        .text,"xr",discard,"#func$hp_target"
; CHECK-NEXT:      .globl  "#func$hp_target"               // -- Begin function #func$hp_target
; CHECK-NEXT:      .p2align        2
; CHECK-NEXT:  "#func$hp_target":                      // @"#func$hp_target"
; CHECK-NEXT:      // %bb.0:
; CHECK-NEXT:      adrp x0, func
; CHECK-NEXT:      add x0, x0, :lo12:func
; CHECK-NEXT:      ret
  ret ptr @func
}

define void @has_varargs(...) hybrid_patchable nounwind {
; SYM: [11](sec  5)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #has_varargs$hp_target
; CHECK-LABEL:     .def "#has_varargs$hp_target";
; CHECK:           .section .text,"xr",discard,"#has_varargs$hp_target"
; CHECK-NEXT:      .globl  "#has_varargs$hp_target"        // -- Begin function #has_varargs$hp_target
; CHECK-NEXT:      .p2align 2
; CHECK-NEXT:  "#has_varargs$hp_target":               // @"#has_varargs$hp_target"
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:      sub sp, sp, #48
; CHECK-NEXT:      stp x0, x1, [x4, #-32]!
; CHECK-NEXT:      stp x2, x3, [x4, #16]
; CHECK-NEXT:      str x4, [sp, #8]
; CHECK-NEXT:      add sp, sp, #48
; CHECK-NEXT:      ret
  %valist = alloca ptr
  call void @llvm.va_start(ptr %valist)
  call void @llvm.va_end(ptr %valist)
  ret void
}

define void @has_sret(ptr sret([100 x i8])) hybrid_patchable nounwind {
; SYM: [14](sec  6)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #has_sret$hp_target
; CHECK-LABEL:     .def    "#has_sret$hp_target";
; CHECK:           .section        .text,"xr",discard,"#has_sret$hp_target"
; CHECK-NEXT:      .globl  "#has_sret$hp_target"           // -- Begin function #has_sret$hp_target
; CHECK-NEXT:      .p2align        2
; CHECK-NEXT:  "#has_sret$hp_target":                  // @"#has_sret$hp_target"
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:      ret
  ret void
}

define dllexport void @exp() hybrid_patchable nounwind {
; CHECK-LABEL:     .def    "#exp$hp_target";
; CHECK:           .section        .text,"xr",discard,"#exp$hp_target"
; CHECK-NEXT:      .globl  "#exp$hp_target"                // -- Begin function #exp$hp_target
; CHECK-NEXT:      .p2align        2
; CHECK-NEXT:  "#exp$hp_target":                       // @"#exp$hp_target"
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:      ret
  ret void
}

; hybrid_patchable attribute is ignored on internal functions
define internal i32 @static_func() hybrid_patchable nounwind {
; CHECK-LABEL:     .def    static_func;
; CHECK:       static_func:                            // @static_func
; CHECK-NEXT:      // %bb.0:
; CHECK-NEXT:      mov     w0, #2                          // =0x2
; CHECK-NEXT:      ret
  ret i32 2
}

define dso_local void @caller() nounwind {
; CHECK-LABEL:     .def    "#caller";
; CHECK:           .section        .text,"xr",discard,"#caller"
; CHECK-NEXT:      .globl  "#caller"                       // -- Begin function #caller
; CHECK-NEXT:      .p2align        2
; CHECK-NEXT:  "#caller":                              // @"#caller"
; CHECK-NEXT:      .weak_anti_dep  caller
; CHECK-NEXT:  caller = "#caller"{{$}}
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:      str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:      bl      "#func"
; CHECK-NEXT:      bl      static_func
; CHECK-NEXT:      adrp    x8, __os_arm64x_check_icall
; CHECK-NEXT:      adrp    x11, func
; CHECK-NEXT:      add     x11, x11, :lo12:func
; CHECK-NEXT:      ldr     x8, [x8, :lo12:__os_arm64x_check_icall]
; CHECK-NEXT:      adrp    x10, $iexit_thunk$cdecl$v$v
; CHECK-NEXT:      add     x10, x10, :lo12:$iexit_thunk$cdecl$v$v
; CHECK-NEXT:      str     x11, [sp, #8]
; CHECK-NEXT:      blr     x8
; CHECK-NEXT:      blr     x11
; CHECK-NEXT:      ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:      ret
  %1 = call i32 @func()
  %2 = call i32 @static_func()
  %3 = alloca ptr, align 8
  store ptr @func, ptr %3, align 8
  %4 = load ptr, ptr %3, align 8
  call void %4()
  ret void
}

; CHECK-LABEL:       def    "#func$hybpatch_thunk";
; CHECK:            .section        .wowthk$aa,"xr",discard,"#func$hybpatch_thunk"
; CHECK-NEXT:       .globl  "#func$hybpatch_thunk"          // -- Begin function #func$hybpatch_thunk
; CHECK-NEXT:       .p2align        2
; CHECK-NEXT:   "#func$hybpatch_thunk":                 // @"#func$hybpatch_thunk"
; CHECK-NEXT:   .seh_proc "#func$hybpatch_thunk"
; CHECK-NEXT:   // %bb.0:
; CHECK-NEXT:       str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:       .seh_save_reg_x x30, 16
; CHECK-NEXT:       .seh_endprologue
; CHECK-NEXT:       adrp    x8, __os_arm64x_dispatch_call
; CHECK-NEXT:       adrp    x11, func
; CHECK-NEXT:       add     x11, x11, :lo12:func
; CHECK-NEXT:       ldr     x8, [x8, :lo12:__os_arm64x_dispatch_call]
; CHECK-NEXT:       adrp    x10, $iexit_thunk$cdecl$i8$v
; CHECK-NEXT:       add     x10, x10, :lo12:$iexit_thunk$cdecl$i8$v
; CHECK-NEXT:       adrp    x9, "#func$hp_target"
; CHECK-NEXT:       add     x9, x9, :lo12:"#func$hp_target"
; CHECK-NEXT:       blr     x8
; CHECK-NEXT:       .seh_startepilogue
; CHECK-NEXT:       ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:       .seh_save_reg_x x30, 16
; CHECK-NEXT:       .seh_endepilogue
; CHECK-NEXT:       br      x11
; CHECK-NEXT:       .seh_endfunclet
; CHECK-NEXT:       .seh_endproc

; CHECK-LABEL:      .def    "#has_varargs$hybpatch_thunk";
; CHECK:            .section        .wowthk$aa,"xr",discard,"#has_varargs$hybpatch_thunk"
; CHECK-NEXT:       .globl  "#has_varargs$hybpatch_thunk"   // -- Begin function #has_varargs$hybpatch_thunk
; CHECK-NEXT:       .p2align        2
; CHECK-NEXT:"#has_varargs$hybpatch_thunk":          // @"#has_varargs$hybpatch_thunk"
; CHECK-NEXT:.seh_proc "#has_varargs$hybpatch_thunk"
; CHECK-NEXT:// %bb.0:
; CHECK-NEXT:       str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:       .seh_save_reg_x x30, 16
; CHECK-NEXT:       .seh_endprologue
; CHECK-NEXT:       adrp    x8, __os_arm64x_dispatch_call
; CHECK-NEXT:       adrp    x11, has_varargs
; CHECK-NEXT:       add     x11, x11, :lo12:has_varargs
; CHECK-NEXT:       ldr     x8, [x8, :lo12:__os_arm64x_dispatch_call]
; CHECK-NEXT:       adrp    x10, $iexit_thunk$cdecl$v$varargs
; CHECK-NEXT:       add     x10, x10, :lo12:$iexit_thunk$cdecl$v$varargs
; CHECK-NEXT:       adrp    x9, "#has_varargs$hp_target"
; CHECK-NEXT:       add     x9, x9, :lo12:"#has_varargs$hp_target"
; CHECK-NEXT:       blr     x8
; CHECK-NEXT:       .seh_startepilogue
; CHECK-NEXT:       ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:       .seh_save_reg_x x30, 16
; CHECK-NEXT:       .seh_endepilogue
; CHECK-NEXT:       br      x11
; CHECK-NEXT:       .seh_endfunclet
; CHECK-NEXT:       .seh_endproc

; CHECK-LABEL:     .def    "#has_sret$hybpatch_thunk";
; CHECK:           .section        .wowthk$aa,"xr",discard,"#has_sret$hybpatch_thunk"
; CHECK-NEXT:      .globl  "#has_sret$hybpatch_thunk"      // -- Begin function #has_sret$hybpatch_thunk
; CHECK-NEXT:      .p2align        2
; CHECK-NEXT:  "#has_sret$hybpatch_thunk":             // @"#has_sret$hybpatch_thunk"
; CHECK-NEXT:  .seh_proc "#has_sret$hybpatch_thunk"
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:      str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:      .seh_save_reg_x x30, 16
; CHECK-NEXT:      .seh_endprologue
; CHECK-NEXT:      adrp    x9, __os_arm64x_dispatch_call
; CHECK-NEXT:      adrp    x11, has_sret
; CHECK-NEXT:      add     x11, x11, :lo12:has_sret
; CHECK-NEXT:      ldr     x12, [x9, :lo12:__os_arm64x_dispatch_call]
; CHECK-NEXT:      adrp    x10, $iexit_thunk$cdecl$m100$v
; CHECK-NEXT:      add     x10, x10, :lo12:$iexit_thunk$cdecl$m100$v
; CHECK-NEXT:      adrp    x9, "#has_sret$hp_target"
; CHECK-NEXT:      add     x9, x9, :lo12:"#has_sret$hp_target"
; CHECK-NEXT:      blr     x12
; CHECK-NEXT:      .seh_startepilogue
; CHECK-NEXT:      ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:      .seh_save_reg_x x30, 16
; CHECK-NEXT:      .seh_endepilogue
; CHECK-NEXT:      br      x11
; CHECK-NEXT:      .seh_endfunclet
; CHECK-NEXT:      .seh_endproc

; CHECK-LABEL:     .def    "#exp$hybpatch_thunk";
; CHECK:           .section        .wowthk$aa,"xr",discard,"#exp$hybpatch_thunk"
; CHECK-NEXT:      .globl  "#exp$hybpatch_thunk"           // -- Begin function #exp$hybpatch_thunk
; CHECK-NEXT:      .p2align        2
; CHECK-NEXT:  "#exp$hybpatch_thunk":                // @"#exp$hybpatch_thunk"
; CHECK-NEXT:  .seh_proc "#exp$hybpatch_thunk"
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:      str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:      .seh_save_reg_x x30, 16
; CHECK-NEXT:      .seh_endprologue
; CHECK-NEXT:      adrp    x8, __os_arm64x_dispatch_call
; CHECK-NEXT:      adrp    x11, exp
; CHECK-NEXT:      add     x11, x11, :lo12:exp
; CHECK-NEXT:      ldr     x8, [x8, :lo12:__os_arm64x_dispatch_call]
; CHECK-NEXT:      adrp    x10, $iexit_thunk$cdecl$v$v
; CHECK-NEXT:      add     x10, x10, :lo12:$iexit_thunk$cdecl$v$v
; CHECK-NEXT:      adrp    x9, "#exp$hp_target"
; CHECK-NEXT:      add     x9, x9, :lo12:"#exp$hp_target"
; CHECK-NEXT:      blr     x8
; CHECK-NEXT:      .seh_startepilogue
; CHECK-NEXT:      ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:      .seh_save_reg_x x30, 16
; CHECK-NEXT:      .seh_endepilogue
; CHECK-NEXT:      br      x11
; CHECK-NEXT:      .seh_endfunclet
; CHECK-NEXT:      .seh_endproc

; Verify the hybrid bitmap
; CHECK-LABEL:     .section        .hybmp$x,"yi"
; CHECK-NEXT:      .symidx "#func$hp_target"
; CHECK-NEXT:      .symidx $ientry_thunk$cdecl$i8$v
; CHECK-NEXT:      .word   1
; CHECK-NEXT:      .symidx "#has_varargs$hp_target"
; CHECK-NEXT:      .symidx $ientry_thunk$cdecl$v$varargs
; CHECK-NEXT:      .word   1
; CHECK-NEXT:      .symidx "#has_sret$hp_target"
; CHECK-NEXT:      .symidx $ientry_thunk$cdecl$m100$v
; CHECK-NEXT:      .word   1
; CHECK-NEXT:      .symidx "#exp$hp_target"
; CHECK-NEXT:      .symidx $ientry_thunk$cdecl$v$v
; CHECK-NEXT:      .word   1
; CHECK-NEXT:      .symidx "#caller"
; CHECK-NEXT:      .symidx $ientry_thunk$cdecl$v$v
; CHECK-NEXT:      .word   1
; CHECK-NEXT:      .symidx func
; CHECK-NEXT:      .symidx $iexit_thunk$cdecl$i8$v
; CHECK-NEXT:      .word   4
; CHECK-NEXT:      .symidx "#func$hybpatch_thunk"
; CHECK-NEXT:      .symidx func
; CHECK-NEXT:      .word   0
; CHECK-NEXT:      .symidx "#has_varargs$hybpatch_thunk"
; CHECK-NEXT:      .symidx has_varargs
; CHECK-NEXT:      .word   0
; CHECK-NEXT:      .symidx "#has_sret$hybpatch_thunk"
; CHECK-NEXT:      .symidx has_sret
; CHECK-NEXT:      .word   0
; CHECK-NEXT:      .symidx "#exp$hybpatch_thunk"
; CHECK-NEXT:      .symidx exp
; CHECK-NEXT:      .word   0
; CHECK-NEXT:      .section        .drectve,"yni"
; CHECK-NEXT:      .ascii  " /EXPORT:exp"

; CHECK-NEXT:      .def    "EXP+#func";
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:      .def    func;
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:      .weak  func
; CHECK-NEXT:  func = "EXP+#func"{{$}}
; CHECK-NEXT:      .weak  "#func"
; CHECK-NEXT:      .def    "#func";
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:  "#func" = "#func$hybpatch_thunk"{{$}}
; CHECK-NEXT:      .def    "EXP+#has_varargs";
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:      .def    has_varargs;
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:      .weak   has_varargs
; CHECK-NEXT:  has_varargs = "EXP+#has_varargs"
; CHECK-NEXT:      .weak   "#has_varargs"
; CHECK-NEXT:      .def    "#has_varargs";
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:  "#has_varargs" = "#has_varargs$hybpatch_thunk"
; CHECK-NEXT:      .def    "EXP+#has_sret";
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:      .def    has_sret;
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:      .weak   has_sret
; CHECK-NEXT:  has_sret = "EXP+#has_sret"
; CHECK-NEXT:      .weak   "#has_sret"
; CHECK-NEXT:      .def    "#has_sret";
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:  "#has_sret" = "#has_sret$hybpatch_thunk"
; CHECK-NEXT:      .def    "EXP+#exp";
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:      .def    exp;
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:      .weak   exp
; CHECK-NEXT:  exp = "EXP+#exp"
; CHECK-NEXT:      .weak   "#exp"
; CHECK-NEXT:      .def    "#exp";
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:  "#exp" = "#exp$hybpatch_thunk"

; SYM:      [53](sec 15)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #func$hybpatch_thunk
; SYM:      [58](sec 16)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #has_varargs$hybpatch_thunk
; SYM:      [68](sec 18)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #has_sret$hybpatch_thunk
; SYM:      [78](sec 20)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #exp$hybpatch_thunk
; SYM:      [110](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 func
; SYM-NEXT: AUX indx 112 srch 3
; SYM-NEXT: [112](sec  0)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 EXP+#func
; SYM:      [116](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 #func
; SYM-NEXT: AUX indx 53 srch 3
; SYM:      [122](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 has_varargs
; SYM-NEXT: AUX indx 124 srch 3
; SYM-NEXT: [124](sec  0)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 EXP+#has_varargs
; SYM-NEXT: [125](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 has_sret
; SYM-NEXT: AUX indx 127 srch 3
; SYM-NEXT: [127](sec  0)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 EXP+#has_sret
; SYM-NEXT: [128](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 exp
; SYM-NEXT: AUX indx 130 srch 3
; SYM-NEXT: [130](sec  0)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 EXP+#exp
; SYM-NEXT: [131](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 #has_varargs
; SYM-NEXT: AUX indx 58 srch 3
; SYM-NEXT: [133](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 #has_sret
; SYM-NEXT: AUX indx 68 srch 3
; SYM-NEXT: [135](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 #exp
; SYM-NEXT: AUX indx 78 srch 3

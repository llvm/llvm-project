; RUN: llc -enable-machine-outliner -verify-machineinstrs -mtriple=armv7-- \
; RUN: -stop-after=machine-outliner < %s | FileCheck %s --check-prefix=ARM
; RUN: llc -enable-machine-outliner -verify-machineinstrs -mtriple=thumbv7-- \
; RUN: -stop-after=machine-outliner < %s | FileCheck %s --check-prefix=THUMB
; RUN: llc -enable-machine-outliner -verify-machineinstrs \
; RUN: -mtriple=thumbv7-apple-darwin -stop-after=machine-outliner < %s \
; RUN: | FileCheck %s --check-prefix=MACHO
; RUN: llc -enable-machine-outliner -verify-machineinstrs -mtriple=thumbv5-- \
; RUN: --stop-after=machine-outliner < %s | FileCheck %s --check-prefix=THUMB1

declare i32 @thunk_called_fn(i32, i32, i32, i32)

define i32 @a() {
; ARM-LABEL: name:             a
; ARM:       bb.0.entry:
; ARM-NEXT:    liveins: $r11, $lr
; ARM:         $sp = frame-setup STMDB_UPD $sp, 14 /* CC::al */, $noreg, killed $r11, killed $lr
; ARM-NEXT:    frame-setup CFI_INSTRUCTION def_cfa_offset 8
; ARM-NEXT:    frame-setup CFI_INSTRUCTION offset $lr, -4
; ARM-NEXT:    frame-setup CFI_INSTRUCTION offset $r11, -8
; ARM-NEXT:    BL @OUTLINED_FUNCTION_0{{.*}}
; ARM-NEXT:    renamable $r0 = ADDri killed renamable $r0, 8, 14 /* CC::al */, $noreg, $noreg
; ARM-NEXT:    $sp = frame-destroy LDMIA_RET $sp, 14 /* CC::al */, $noreg, def $r11, def $pc, implicit killed $r0

; THUMB-LABEL: name:             a
; THUMB:       bb.0.entry:
; THUMB-NEXT:    liveins: $r7, $lr
; THUMB:         frame-setup tPUSH 14 /* CC::al */, $noreg, killed $r7, killed $lr
; THUMB-NEXT:    frame-setup CFI_INSTRUCTION def_cfa_offset 8
; THUMB-NEXT:    frame-setup CFI_INSTRUCTION offset $lr, -4
; THUMB-NEXT:    frame-setup CFI_INSTRUCTION offset $r7, -8
; THUMB-NEXT:    tBL 14 /* CC::al */, $noreg, @OUTLINED_FUNCTION_0{{.*}}
; THUMB-NEXT:    renamable $r0, dead $cpsr = tADDi8 killed renamable $r0, 8, 14 /* CC::al */, $noreg
; THUMB-NEXT:    tPOP_RET 14 /* CC::al */, $noreg, def $r7, def $pc

; MACHO-LABEL: name:             a
; MACHO:       bb.0.entry:
; MACHO-NEXT:    liveins: $lr
; MACHO:         early-clobber $sp = frame-setup t2STR_PRE killed $lr, $sp, -4, 14 /* CC::al */, $noreg
; MACHO-NEXT:    frame-setup CFI_INSTRUCTION def_cfa_offset 4
; MACHO-NEXT:    frame-setup CFI_INSTRUCTION offset $lr, -4
; MACHO-NEXT:    tBL 14 /* CC::al */, $noreg, @OUTLINED_FUNCTION_0{{.*}}
; MACHO-NEXT:    renamable $r0, dead $cpsr = tADDi8 killed renamable $r0, 8, 14 /* CC::al */, $noreg
; MACHO-NEXT:    $lr, $sp = frame-destroy t2LDR_POST $sp, 4, 14 /* CC::al */, $noreg
; MACHO-NEXT:    tBX_RET 14 /* CC::al */, $noreg, implicit killed $r0

; THUMB1-NOT: OUTLINED_FUNCTION_0

entry:
  %call = tail call i32 @thunk_called_fn(i32 1, i32 2, i32 3, i32 4)
  %cx = add i32 %call, 8
  ret i32 %cx
}

define i32 @b() {
; ARM-LABEL: name:             b
; ARM:       bb.0.entry:
; ARM-NEXT:    liveins: $r11, $lr
; ARM:         $sp = frame-setup STMDB_UPD $sp, 14 /* CC::al */, $noreg, killed $r11, killed $lr
; ARM-NEXT:    frame-setup CFI_INSTRUCTION def_cfa_offset 8
; ARM-NEXT:    frame-setup CFI_INSTRUCTION offset $lr, -4
; ARM-NEXT:    frame-setup CFI_INSTRUCTION offset $r11, -8
; ARM-NEXT:    BL @OUTLINED_FUNCTION_0{{.*}}
; ARM-NEXT:    renamable $r0 = ADDri killed renamable $r0, 88, 14 /* CC::al */, $noreg, $noreg
; ARM-NEXT:    $sp = frame-destroy LDMIA_RET $sp, 14 /* CC::al */, $noreg, def $r11, def $pc, implicit killed $r0

; THUMB-LABEL: name:             b
; THUMB:       bb.0.entry:
; THUMB-NEXT:    liveins: $r7, $lr
; THUMB:         frame-setup tPUSH 14 /* CC::al */, $noreg, killed $r7, killed $lr
; THUMB-NEXT:    frame-setup CFI_INSTRUCTION def_cfa_offset 8
; THUMB-NEXT:    frame-setup CFI_INSTRUCTION offset $lr, -4
; THUMB-NEXT:    frame-setup CFI_INSTRUCTION offset $r7, -8
; THUMB-NEXT:    tBL 14 /* CC::al */, $noreg, @OUTLINED_FUNCTION_0{{.*}}
; THUMB-NEXT:    renamable $r0, dead $cpsr = tADDi8 killed renamable $r0, 88, 14 /* CC::al */, $noreg
; THUMB-NEXT:    tPOP_RET 14 /* CC::al */, $noreg, def $r7, def $pc

; MACHO-LABEL: name:             b
; MACHO:       bb.0.entry:
; MACHO-NEXT:    liveins: $lr
; MACHO:         early-clobber $sp = frame-setup t2STR_PRE killed $lr, $sp, -4, 14 /* CC::al */, $noreg
; MACHO-NEXT:    frame-setup CFI_INSTRUCTION def_cfa_offset 4
; MACHO-NEXT:    frame-setup CFI_INSTRUCTION offset $lr, -4
; MACHO-NEXT:    tBL 14 /* CC::al */, $noreg, @OUTLINED_FUNCTION_0{{.*}}
; MACHO-NEXT:    renamable $r0, dead $cpsr = tADDi8 killed renamable $r0, 88, 14 /* CC::al */, $noreg
; MACHO-NEXT:    $lr, $sp = frame-destroy t2LDR_POST $sp, 4, 14 /* CC::al */, $noreg
; MACHO-NEXT:    tBX_RET 14 /* CC::al */, $noreg, implicit killed $r0
entry:
  %call = tail call i32 @thunk_called_fn(i32 1, i32 2, i32 3, i32 4)
  %cx = add i32 %call, 88
  ret i32 %cx
}

; ARM-LABEL: name:            OUTLINED_FUNCTION_0
; ARM:        bb.0:
; ARM-NEXT:   liveins: $r10, $r9, $r8, $r7, $r6, $r5, $r4, $d15, $d14, $d13, $d12, $d11, $d10, $d9, $d8
; ARM:        $r0 = MOVi 1, 14 /* CC::al */, $noreg, $noreg
; ARM-NEXT:   $r1 = MOVi 2, 14 /* CC::al */, $noreg, $noreg
; ARM-NEXT:   $r2 = MOVi 3, 14 /* CC::al */, $noreg, $noreg
; ARM-NEXT:   $r3 = MOVi 4, 14 /* CC::al */, $noreg, $noreg
; ARM-NEXT:   TAILJMPd @thunk_called_fn, implicit $sp

; THUMB-LABEL: name:            OUTLINED_FUNCTION_0
; THUMB:        bb.0:
; THUMB-NEXT:   liveins: $r11, $r10, $r9, $r8, $r6, $r5, $r4, $d15, $d14, $d13, $d12, $d11, $d10, $d9, $d8
; THUMB:        $r0, dead $cpsr = tMOVi8 1, 14 /* CC::al */, $noreg
; THUMB-NEXT:   $r1, dead $cpsr = tMOVi8 2, 14 /* CC::al */, $noreg
; THUMB-NEXT:   $r2, dead $cpsr = tMOVi8 3, 14 /* CC::al */, $noreg
; THUMB-NEXT:   $r3, dead $cpsr = tMOVi8 4, 14 /* CC::al */, $noreg
; THUMB-NEXT:   tTAILJMPdND @thunk_called_fn, 14 /* CC::al */, $noreg, implicit $sp

; MACHO-LABEL: name:            OUTLINED_FUNCTION_0
; MACHO:        bb.0:
; MACHO-NEXT:   liveins: $r7, $r6, $r5, $r4, $r11, $r10, $r8, $d15, $d14, $d13, $d12, $d11, $d10, $d9, $d8
; MACHO:        $r0, dead $cpsr = tMOVi8 1, 14 /* CC::al */, $noreg
; MACHO-NEXT:   $r1, dead $cpsr = tMOVi8 2, 14 /* CC::al */, $noreg
; MACHO-NEXT:   $r2, dead $cpsr = tMOVi8 3, 14 /* CC::al */, $noreg
; MACHO-NEXT:   $r3, dead $cpsr = tMOVi8 4, 14 /* CC::al */, $noreg
; MACHO-NEXT:   tTAILJMPd @thunk_called_fn, 14 /* CC::al */, $noreg, implicit $sp

; RUN: llc -enable-machine-outliner -verify-machineinstrs -mtriple=arm-- \
; RUN: --stop-after=machine-outliner < %s | FileCheck %s --check-prefix=ARM
; RUN: llc -enable-machine-outliner -verify-machineinstrs -mtriple=thumbv7-- \
; RUN: --stop-after=machine-outliner < %s | FileCheck %s --check-prefix=THUMB
; RUN: llc -enable-machine-outliner -verify-machineinstrs \
; RUN: -mtriple=thumbv7-apple-darwin --stop-after=machine-outliner < %s \
; RUN: | FileCheck %s --check-prefix=MACHO
; RUN: llc -enable-machine-outliner -verify-machineinstrs -mtriple=thumbv5-- \
; RUN: --stop-after=machine-outliner < %s | FileCheck %s --check-prefix=THUMB1
; RUN: llc -verify-machineinstrs -mtriple=thumbv8m.main \
; RUN: --stop-after=machine-outliner < %s | FileCheck %s --check-prefix=THUMB

; ARM-LABEL: name:            OUTLINED_FUNCTION_0
; ARM: $r0 = MOVi 1, 14 /* CC::al */, $noreg, $noreg
; ARM-NEXT: $r1 = MOVi 2, 14 /* CC::al */, $noreg, $noreg
; ARM-NEXT: $r2 = MOVi 3, 14 /* CC::al */, $noreg, $noreg
; ARM-NEXT: $r3 = MOVi 4, 14 /* CC::al */, $noreg, $noreg
; ARM-NEXT: TAILJMPd @z

; THUMB-LABEL: name:            OUTLINED_FUNCTION_0
; THUMB: $r0, dead $cpsr = tMOVi8 1, 14 /* CC::al */, $noreg
; THUMB-NEXT: $r1, dead $cpsr = tMOVi8 2, 14 /* CC::al */, $noreg
; THUMB-NEXT: $r2, dead $cpsr = tMOVi8 3, 14 /* CC::al */, $noreg
; THUMB-NEXT: $r3, dead $cpsr = tMOVi8 4, 14 /* CC::al */, $noreg
; THUMB-NEXT: tTAILJMPdND @z, 14 /* CC::al */, $noreg

; MACHO-LABEL: name:            OUTLINED_FUNCTION_0
; MACHO: $r0, dead $cpsr = tMOVi8 1, 14 /* CC::al */, $noreg
; MACHO-NEXT: $r1, dead $cpsr = tMOVi8 2, 14 /* CC::al */, $noreg
; MACHO-NEXT: $r2, dead $cpsr = tMOVi8 3, 14 /* CC::al */, $noreg
; MACHO-NEXT: $r3, dead $cpsr = tMOVi8 4, 14 /* CC::al */, $noreg
; MACHO-NEXT: tTAILJMPd @z, 14 /* CC::al */, $noreg

; THUMB1-NOT: OUTLINED_FUNCTION_0

define void @a() #0 {
entry:
  tail call void @z(i32 1, i32 2, i32 3, i32 4)
  ret void
}

declare void @z(i32, i32, i32, i32)

define dso_local void @b(ptr nocapture readnone %p) #0 {
entry:
  tail call void @z(i32 1, i32 2, i32 3, i32 4)
  ret void
}

attributes #0 = { minsize optsize }

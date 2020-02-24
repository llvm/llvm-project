; RUN: llc -mtriple=powerpc64-unknown-aix-xcoff -mcpu=pwr4 --mattr=-altivec \
; RUN: --verify-machineinstrs < %s | FileCheck --check-prefix=64BIT %s

; RUN: llc -mtriple=powerpc-unknown-aix-xcoff -mcpu=pwr4 --mattr=-altivec \
; RUN: --verify-machineinstrs < %s | FileCheck --check-prefix=32BIT %s

define dso_local signext i32 @killOne(i32 signext %i) {
entry:
  tail call void asm sideeffect "# Clobber CR", "~{cr4}"()
  %call = call signext i32 @do_something(i32 %i)
  ret i32 %call
}

define dso_local signext i32 @killAll(i32 signext %i) {
entry:
  tail call void asm sideeffect "# Clobber CR", "~{cr0},~{cr1},~{cr2},~{cr3},~{cr4},~{cr5},~{cr6},~{cr7}" ()
  %call = call signext i32 @do_something(i32 %i)
  ret i32 %call
}

declare signext i32 @do_something(i32 signext)

; 64BIT-LABEL: .killOne:

; 64BIT:       mflr 0
; 64BIT-NEXT:  std 0, 16(1)
; 64BIT-NEXT:  mfcr 12
; 64BIT-NEXT:  stw 12, 8(1)
; 64BIT:       stdu 1, -112(1)

; 64BIT:       # Clobber CR
; 64BIT:       bl .do_something

; 64BIT:       addi 1, 1, 112
; 64BIT-NEXT:  ld 0, 16(1)
; 64BIT-NEXT:  lwz 12, 8(1)
; 64BIT-NEXT:  mtlr 0
; 64BIT-NEXT:  mtocrf 8, 12
; 64BIT:       blr

; 32BIT-LABEL: .killOne:

; 32BIT:       mflr 0
; 32BIT-NEXT:  stw 0, 8(1)
; 32BIT-NEXT:  mfcr 12
; 32BIT-NEXT:  stw 12, 4(1)
; 32BIT:       stwu 1, -64(1)

; 32BIT:       # Clobber CR
; 32BIT:       bl .do_something

; 32BIT:       addi 1, 1, 64
; 32BIT-NEXT:  lwz 0, 8(1)
; 32BIT-NEXT:  lwz 12, 4(1)
; 32BIT-NEXT:  mtlr 0
; 32BIT-NEXT:  mtocrf 8, 12
; 32BIT:       blr


; 64BIT-LABEL: .killAll:

; 64BIT:        addi 1, 1, 112
; 64BIT-NEXT:   ld 0, 16(1)
; 64BIT-NEXT:   lwz 12, 8(1)
; 64BIT-NEXT:   mtlr 0
; 64BIT-NEXT:   mtocrf 32, 12
; 64BIT-NEXT:   mtocrf 16, 12
; 64BIT-NEXT:   mtocrf 8, 12
; 64BIT-NEXT:   blr


; 32BIT-LABEL: .killAll:

; 32BIT:        addi 1, 1, 64
; 32BIT-NEXT:   lwz 0, 8(1)
; 32BIT-NEXT:   lwz 12, 4(1)
; 32BIT-NEXT:   mtlr 0
; 32BIT-NEXT:   mtocrf 32, 12
; 32BIT-NEXT:   mtocrf 16, 12
; 32BIT-NEXT:   mtocrf 8, 12
; 32BIT-NEXT:   blr

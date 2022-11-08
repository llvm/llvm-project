; RUN: llc -mtriple=powerpc64-unknown-aix-xcoff -verify-machineinstrs \
; RUN: -mcpu=pwr4 -mattr=-altivec -stop-after=prologepilog < %s | \
; RUN: FileCheck --check-prefix=MIR64 %s

; RUN: llc -mtriple=powerpc64-unknown-aix-xcoff -verify-machineinstrs \
; RUN: -mcpu=pwr4 -mattr=-altivec < %s | FileCheck --check-prefix=ASM64 %s

; RUN: llc -mtriple=powerpc-unknown-aix-xcoff -verify-machineinstrs \
; RUN: -mcpu=pwr4 -mattr=-altivec -stop-after=prologepilog < %s | \
; RUN: FileCheck --check-prefix=MIR32 %s

; RUN: llc -mtriple=powerpc-unknown-aix-xcoff -verify-machineinstrs \
; RUN: -mcpu=pwr4 -mattr=-altivec < %s | FileCheck --check-prefix=ASM32 %s

define dso_local signext i32 @gprs_only(i32 signext %i) {
entry:
  call void asm sideeffect "", "~{r16},~{r22},~{r30}"()
  ret i32 %i
}

; MIR64:       name:            gprs_only
; MIR64-LABEL: fixedStack:
; MIR64-NEXT:   - { id: 0, type: spill-slot, offset: -16, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:       callee-saved-register: '$x30', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:       debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:   - { id: 1, type: spill-slot, offset: -80, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:       callee-saved-register: '$x22', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:       debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:   - { id: 2, type: spill-slot, offset: -128, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:       callee-saved-register: '$x16', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:       debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  stack:           []

; MIR32:       name:            gprs_only
; MIR32-LABEL: fixedStack:
; MIR32:        - { id: 0, type: spill-slot, offset: -8, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:       callee-saved-register: '$r30', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:       debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:   - { id: 1, type: spill-slot, offset: -40, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:       callee-saved-register: '$r22', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:       debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:   - { id: 2, type: spill-slot, offset: -64, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:       callee-saved-register: '$r16', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:       debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  stack:           []


; MIR64: liveins: $x3, $x16, $x22, $x30

; MIR64-DAG: STD killed $x16, -128, $x1 :: (store (s64) into %fixed-stack.2, align 16)
; MIR64-DAG: STD killed $x22, -80, $x1 :: (store (s64) into %fixed-stack.1, align 16)
; MIR64-DAG: STD killed $x30, -16, $x1 :: (store (s64) into %fixed-stack.0, align 16)

; MIR64:     INLINEASM

; MIR64-DAG: $x30 = LD -16, $x1 :: (load (s64) from %fixed-stack.0, align 16)
; MIR64-DAG: $x22 = LD -80, $x1 :: (load (s64) from %fixed-stack.1, align 16)
; MIR64-DAG: $x16 = LD -128, $x1 :: (load (s64) from %fixed-stack.2, align 16)
; MIR64:     BLR8 implicit $lr8, implicit $rm, implicit $x3


; MIR32: liveins: $r3, $r16, $r22, $r30

; MIR32-DAG: STW killed $r16, -64, $r1 :: (store (s32) into %fixed-stack.2, align 16)
; MIR32-DAG: STW killed $r22, -40, $r1 :: (store (s32) into %fixed-stack.1, align 8)
; MIR32-DAG: STW killed $r30, -8, $r1 :: (store (s32) into %fixed-stack.0, align 8)

; MIR32:     INLINEASM

; MIR32-DAG: $r30 = LWZ -8, $r1 :: (load (s32) from %fixed-stack.0, align 8)
; MIR32-DAG: $r22 = LWZ -40, $r1 :: (load (s32) from %fixed-stack.1, align 8)
; MIR32-DAG: $r16 = LWZ -64, $r1 :: (load (s32) from %fixed-stack.2, align 16)
; MIR32:     BLR implicit $lr, implicit $rm, implicit $r3


; ASM64-LABEL: .gprs_only:
; ASM64-DAG:      std 16, -128(1)                 # 8-byte Folded Spill
; ASM64-DAG:      std 22, -80(1)                  # 8-byte Folded Spill
; ASM64-DAG:      std 30, -16(1)                  # 8-byte Folded Spill
; ASM64:          #APP
; ASM64-DAG:      ld 30, -16(1)                   # 8-byte Folded Reload
; ASM64-DAG:      ld 22, -80(1)                   # 8-byte Folded Reload
; ASM64-DAG:      ld 16, -128(1)                  # 8-byte Folded Reload
; ASM64:          blr

; ASM32-LABEL: .gprs_only:
; ASM32-DAG:     stw 16, -64(1)                  # 4-byte Folded Spill
; ASM32-DAG:     stw 22, -40(1)                  # 4-byte Folded Spill
; ASM32-DAG:     stw 30, -8(1)                   # 4-byte Folded Spill
; ASM32:         #APP
; ASM32-DAG:     lwz 30, -8(1)                   # 4-byte Folded Reload
; ASM32-DAG:     lwz 22, -40(1)                  # 4-byte Folded Reload
; ASM32-DAG:     lwz 16, -64(1)                  # 4-byte Folded Reload
; ASM32-DAG:     blr


declare double @dummy(i32 signext);

define dso_local double @fprs_and_gprs(i32 signext %i) {
  call void asm sideeffect "", "~{r13},~{r14},~{r25},~{r31},~{f14},~{f19},~{f21},~{f31}"()
  %result = call double @dummy(i32 signext %i)
  ret double %result
}

; MIR64:       name:            fprs_and_gprs
; MIR64-LABEL: fixedStack:
; MIR64-NEXT:    - { id: 0, type: spill-slot, offset: -8, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$f31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 1, type: spill-slot, offset: -88, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$f21', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 2, type: spill-slot, offset: -104, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$f19', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 3, type: spill-slot, offset: -144, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$f14', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 4, type: spill-slot, offset: -152, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$x31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 5, type: spill-slot, offset: -200, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$x25', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 6, type: spill-slot, offset: -288, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$x14', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  stack:           []

; MIR32:       name:            fprs_and_gprs
; MIR32-LABEL: fixedStack:
; MIR32-NEXT:    - { id: 0, type: spill-slot, offset: -8, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$f31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 1, type: spill-slot, offset: -88, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$f21', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 2, type: spill-slot, offset: -104, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$f19', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 3, type: spill-slot, offset: -144, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$f14', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 4, type: spill-slot, offset: -148, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$r31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 5, type: spill-slot, offset: -172, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$r25', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 6, type: spill-slot, offset: -216, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$r14', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 7, type: spill-slot, offset: -220, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$r13', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  stack:           []


; MIR64: liveins: $x3, $x14, $x25, $x31, $f14, $f19, $f21, $f31

; MIR64:       $x0 = MFLR8 implicit $lr8
; MIR64-NEXT:  $x1 = STDU $x1, -400, $x1
; MIR64-NEXT:  STD killed $x0, 416, $x1
; MIR64-DAG:   STD killed $x14, 112, $x1 :: (store (s64) into %fixed-stack.6, align 16)
; MIR64-DAG:   STD killed $x25, 200, $x1 :: (store (s64) into %fixed-stack.5)
; MIR64-DAG:   STD killed $x31, 248, $x1 :: (store (s64) into %fixed-stack.4)
; MIR64-DAG:   STFD killed $f14, 256, $x1 :: (store (s64) into %fixed-stack.3, align 16)
; MIR64-DAG:   STFD killed $f19, 296, $x1 :: (store (s64) into %fixed-stack.2)
; MIR64-DAG:   STFD killed $f21, 312, $x1 :: (store (s64) into %fixed-stack.1)
; MIR64-DAG:   STFD killed $f31, 392, $x1 :: (store (s64) into %fixed-stack.0)

; MIR64:       INLINEASM
; MIR64-NEXT:  BL8_NOP

; MIR64-DAG:   $f31 = LFD 392, $x1 :: (load (s64) from %fixed-stack.0)
; MIR64-DAG:   $f21 = LFD 312, $x1 :: (load (s64) from %fixed-stack.1)
; MIR64-DAG:   $f19 = LFD 296, $x1 :: (load (s64) from %fixed-stack.2)
; MIR64-DAG:   $f14 = LFD 256, $x1 :: (load (s64) from %fixed-stack.3, align 16)
; MIR64-DAG:   $x31 = LD 248, $x1 :: (load (s64) from %fixed-stack.4)
; MIR64-DAG:   $x25 = LD 200, $x1 :: (load (s64) from %fixed-stack.5)
; MIR64-DAG:   $x14 = LD 112, $x1 :: (load (s64) from %fixed-stack.6, align 16)
; MIR64:       $x1 = ADDI8 $x1, 400
; MIR64-NEXT:  $x0 = LD 16, $x1
; MIR64-NEXT:  MTLR8 $x0, implicit-def $lr8
; MIR64-NEXT:  BLR8 implicit $lr8, implicit $rm, implicit $f1


; MIR32: liveins: $r3, $r13, $r14, $r25, $r31, $f14, $f19, $f21, $f31

; MIR32:      $r0 = MFLR implicit $lr
; MIR32-NEXT: $r1 = STWU $r1, -288, $r1
; MIR32-NEXT: STW killed $r0, 296, $r1
; MIR32-DAG:  STW killed $r13, 68, $r1 :: (store (s32) into %fixed-stack.7)
; MIR32-DAG:  STW killed $r14, 72, $r1 :: (store (s32) into %fixed-stack.6, align 8)
; MIR32-DAG:  STW killed $r25, 116, $r1 :: (store (s32) into %fixed-stack.5)
; MIR32-DAG:  STW killed $r31, 140, $r1 :: (store (s32) into %fixed-stack.4)
; MIR32-DAG:  STFD killed $f14, 144, $r1 :: (store (s64) into %fixed-stack.3, align 16)
; MIR32-DAG:  STFD killed $f19, 184, $r1 :: (store (s64) into %fixed-stack.2)
; MIR32-DAG:  STFD killed $f21, 200, $r1 :: (store (s64) into %fixed-stack.1)
; MIR32-DAG:  STFD killed $f31, 280, $r1 :: (store (s64) into %fixed-stack.0)

; MIR32:      INLINEASM
; MIR32:      BL_NOP

; MIR32-DAG:  $f31 = LFD 280, $r1 :: (load (s64) from %fixed-stack.0)
; MIR32-DAG:  $f21 = LFD 200, $r1 :: (load (s64) from %fixed-stack.1)
; MIR32-DAG:  $f19 = LFD 184, $r1 :: (load (s64) from %fixed-stack.2)
; MIR32-DAG:  $f14 = LFD 144, $r1 :: (load (s64) from %fixed-stack.3, align 16)
; MIR32-DAG:  $r31 = LWZ 140, $r1 :: (load (s32) from %fixed-stack.4)
; MIR32-DAG:  $r25 = LWZ 116, $r1 :: (load (s32) from %fixed-stack.5)
; MIR32-DAG:  $r14 = LWZ 72, $r1 :: (load (s32) from %fixed-stack.6, align 8)
; MIR32-DAG:  $r13 = LWZ 68, $r1 :: (load (s32) from %fixed-stack.7)
; MIR32:      $r1 = ADDI $r1, 288
; MIR32-NEXT: $r0 = LWZ 8, $r1
; MIR32-NEXT: MTLR $r0, implicit-def $lr
; MIR32-NEXT: BLR implicit $lr, implicit $rm, implicit $f1

; ASM64-LABEL: .fprs_and_gprs:
; ASM64:         mflr 0
; ASM64-NEXT:    stdu 1, -400(1)
; ASM64-NEXT:    std 0, 416(1)
; ASM64-DAG:     std 14, 112(1)                  # 8-byte Folded Spill
; ASM64-DAG:     std 25, 200(1)                  # 8-byte Folded Spill
; ASM64-DAG:     std 31, 248(1)                  # 8-byte Folded Spill
; ASM64-DAG:     stfd 14, 256(1)                 # 8-byte Folded Spill
; ASM64-DAG:     stfd 19, 296(1)                 # 8-byte Folded Spill
; ASM64-DAG:     stfd 21, 312(1)                 # 8-byte Folded Spill
; ASM64-DAG:     stfd 31, 392(1)                 # 8-byte Folded Spill

; ASM64:         bl .dummy

; ASM64-DAG:     lfd 31, 392(1)                  # 8-byte Folded Reload
; ASM64-DAG:     lfd 21, 312(1)                  # 8-byte Folded Reload
; ASM64-DAG:     lfd 19, 296(1)                  # 8-byte Folded Reload
; ASM64-DAG:     lfd 14, 256(1)                  # 8-byte Folded Reload
; ASM64-DAG:     ld 31, 248(1)                   # 8-byte Folded Reload
; ASM64-DAG:     ld 25, 200(1)                   # 8-byte Folded Reload
; ASM64-DAG:     ld 14, 112(1)                   # 8-byte Folded Reload
; ASM64:         addi 1, 1, 400
; ASM64-NEXT:    ld 0, 16(1)
; ASM64-NEXT:    mtlr 0
; ASM64-NEXT:    blr

; ASM32-LABEL: .fprs_and_gprs:
; ASM32:         mflr 0
; ASM32-NEXT:    stwu 1, -288(1)
; ASM32-NEXT:    stw 0, 296(1)
; ASM32-DAG:     stw 13, 68(1)                   # 4-byte Folded Spill
; ASM32-DAG:     stw 14, 72(1)                   # 4-byte Folded Spill
; ASM32-DAG:     stw 25, 116(1)                  # 4-byte Folded Spill
; ASM32-DAG:     stw 31, 140(1)                  # 4-byte Folded Spill
; ASM32-DAG:     stfd 14, 144(1)                 # 8-byte Folded Spill
; ASM32-DAG:     stfd 19, 184(1)                 # 8-byte Folded Spill
; ASM32-DAG:     stfd 21, 200(1)                 # 8-byte Folded Spill
; ASM32-DAG:     stfd 31, 280(1)                 # 8-byte Folded Spill

; ASM32-DAG:     bl .dummy

; ASM32-DAG:     lfd 31, 280(1)                  # 8-byte Folded Reload
; ASM32-DAG:     lfd 21, 200(1)                  # 8-byte Folded Reload
; ASM32-DAG:     lfd 19, 184(1)                  # 8-byte Folded Reload
; ASM32-DAG:     lfd 14, 144(1)                  # 8-byte Folded Reload
; ASM32-DAG:     lwz 31, 140(1)                  # 4-byte Folded Reload
; ASM32-DAG:     lwz 25, 116(1)                  # 4-byte Folded Reload
; ASM32-DAG:     lwz 14, 72(1)                   # 4-byte Folded Reload
; ASM32-DAG:     lwz 13, 68(1)                   # 4-byte Folded Reload
; ASM32:         addi 1, 1, 288
; ASM32-NEXT:    lwz 0, 8(1)
; ASM32-NEXT:    mtlr 0
; ASM32-NEXT:    blr

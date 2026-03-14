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
; MIR64-NEXT:  - { id: 0, type: spill-slot, offset: -8, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 1, type: spill-slot, offset: -16, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x30', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 2, type: spill-slot, offset: -24, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x29', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 3, type: spill-slot, offset: -32, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x28', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 4, type: spill-slot, offset: -40, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x27', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 5, type: spill-slot, offset: -48, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x26', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 6, type: spill-slot, offset: -56, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x25', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 7, type: spill-slot, offset: -64, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x24', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 8, type: spill-slot, offset: -72, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x23', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 9, type: spill-slot, offset: -80, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x22', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 10, type: spill-slot, offset: -88, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x21', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 11, type: spill-slot, offset: -96, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x20', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 12, type: spill-slot, offset: -104, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x19', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 13, type: spill-slot, offset: -112, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x18', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 14, type: spill-slot, offset: -120, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x17', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  - { id: 15, type: spill-slot, offset: -128, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:      callee-saved-register: '$x16', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  stack:           []

; MIR32:       name:            gprs_only
; MIR32-LABEL: fixedStack:
; MIR32-NEXT:  - { id: 0, type: spill-slot, offset: -4, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 1, type: spill-slot, offset: -8, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r30', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 2, type: spill-slot, offset: -12, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r29', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 3, type: spill-slot, offset: -16, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r28', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 4, type: spill-slot, offset: -20, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r27', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 5, type: spill-slot, offset: -24, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r26', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 6, type: spill-slot, offset: -28, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r25', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 7, type: spill-slot, offset: -32, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r24', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 8, type: spill-slot, offset: -36, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r23', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 9, type: spill-slot, offset: -40, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r22', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 10, type: spill-slot, offset: -44, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r21', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 11, type: spill-slot, offset: -48, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r20', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 12, type: spill-slot, offset: -52, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r19', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 13, type: spill-slot, offset: -56, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r18', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 14, type: spill-slot, offset: -60, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r17', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 15, type: spill-slot, offset: -64, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r16', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  stack:           []


; MIR64: liveins: $x3, $x16, $x17, $x18, $x19, $x20, $x21, $x22, $x23, $x24, $x25, $x26, $x27, $x28, $x29, $x30, $x31

; MIR64-DAG:       STD killed $x16, -128, $x1 :: (store (s64) into %fixed-stack.15, align 16)
; MIR64-DAG:  STD killed $x17, -120, $x1 :: (store (s64) into %fixed-stack.14)
; MIR64-DAG:  STD killed $x18, -112, $x1 :: (store (s64) into %fixed-stack.13, align 16)
; MIR64-DAG:  STD killed $x19, -104, $x1 :: (store (s64) into %fixed-stack.12)
; MIR64-DAG:  STD killed $x20, -96, $x1 :: (store (s64) into %fixed-stack.11, align 16)
; MIR64-DAG:  STD killed $x21, -88, $x1 :: (store (s64) into %fixed-stack.10)
; MIR64-DAG:  STD killed $x22, -80, $x1 :: (store (s64) into %fixed-stack.9, align 16)
; MIR64-DAG:  STD killed $x23, -72, $x1 :: (store (s64) into %fixed-stack.8)
; MIR64-DAG:  STD killed $x24, -64, $x1 :: (store (s64) into %fixed-stack.7, align 16)
; MIR64-DAG:  STD killed $x25, -56, $x1 :: (store (s64) into %fixed-stack.6)
; MIR64-DAG:  STD killed $x26, -48, $x1 :: (store (s64) into %fixed-stack.5, align 16)
; MIR64-DAG:  STD killed $x27, -40, $x1 :: (store (s64) into %fixed-stack.4)
; MIR64-DAG:  STD killed $x28, -32, $x1 :: (store (s64) into %fixed-stack.3, align 16)
; MIR64-DAG:  STD killed $x29, -24, $x1 :: (store (s64) into %fixed-stack.2)
; MIR64-DAG:  STD killed $x30, -16, $x1 :: (store (s64) into %fixed-stack.1, align 16)
; MIR64-DAG:  STD killed $x31, -8, $x1 :: (store (s64) into %fixed-stack.0)

; MIR64:     INLINEASM


; MIR64-DAG:    $x31 = LD -8, $x1 :: (load (s64) from %fixed-stack.0)
; MIR64-DAG:    $x30 = LD -16, $x1 :: (load (s64) from %fixed-stack.1, align 16)
; MIR64-DAG:    $x29 = LD -24, $x1 :: (load (s64) from %fixed-stack.2)
; MIR64-DAG:    $x28 = LD -32, $x1 :: (load (s64) from %fixed-stack.3, align 16)
; MIR64-DAG:    $x27 = LD -40, $x1 :: (load (s64) from %fixed-stack.4)
; MIR64-DAG:    $x26 = LD -48, $x1 :: (load (s64) from %fixed-stack.5, align 16)
; MIR64-DAG:    $x25 = LD -56, $x1 :: (load (s64) from %fixed-stack.6)
; MIR64-DAG:    $x24 = LD -64, $x1 :: (load (s64) from %fixed-stack.7, align 16)
; MIR64-DAG:    $x23 = LD -72, $x1 :: (load (s64) from %fixed-stack.8)
; MIR64-DAG:    $x22 = LD -80, $x1 :: (load (s64) from %fixed-stack.9, align 16)
; MIR64-DAG:    $x21 = LD -88, $x1 :: (load (s64) from %fixed-stack.10)
; MIR64-DAG:    $x20 = LD -96, $x1 :: (load (s64) from %fixed-stack.11, align 16)
; MIR64-DAG:    $x19 = LD -104, $x1 :: (load (s64) from %fixed-stack.12)
; MIR64-DAG:    $x18 = LD -112, $x1 :: (load (s64) from %fixed-stack.13, align 16)
; MIR64-DAG:    $x17 = LD -120, $x1 :: (load (s64) from %fixed-stack.14)
; MIR64-DAG:    $x16 = LD -128, $x1 :: (load (s64) from %fixed-stack.15, align 16)
; MIR64:        BLR8 implicit $lr8, implicit $rm, implicit $x3


; MIR32:  liveins: $r3, $r16, $r17, $r18, $r19, $r20, $r21, $r22, $r23, $r24, $r25, $r26, $r27, $r28, $r29, $r30, $r31

; MIR32-DAG:  STW killed $r16, -64, $r1 :: (store (s32) into %fixed-stack.15, align 16)
; MIR32-DAG:  STW killed $r17, -60, $r1 :: (store (s32) into %fixed-stack.14)
; MIR32-DAG:  STW killed $r18, -56, $r1 :: (store (s32) into %fixed-stack.13, align 8)
; MIR32-DAG:  STW killed $r19, -52, $r1 :: (store (s32) into %fixed-stack.12)
; MIR32-DAG:  STW killed $r20, -48, $r1 :: (store (s32) into %fixed-stack.11, align 16)
; MIR32-DAG:  STW killed $r21, -44, $r1 :: (store (s32) into %fixed-stack.10)
; MIR32-DAG:  STW killed $r22, -40, $r1 :: (store (s32) into %fixed-stack.9, align 8)
; MIR32-DAG:  STW killed $r23, -36, $r1 :: (store (s32) into %fixed-stack.8)
; MIR32-DAG:  STW killed $r24, -32, $r1 :: (store (s32) into %fixed-stack.7, align 16)
; MIR32-DAG:  STW killed $r25, -28, $r1 :: (store (s32) into %fixed-stack.6)
; MIR32-DAG:  STW killed $r26, -24, $r1 :: (store (s32) into %fixed-stack.5, align 8)
; MIR32-DAG:  STW killed $r27, -20, $r1 :: (store (s32) into %fixed-stack.4)
; MIR32-DAG:  STW killed $r28, -16, $r1 :: (store (s32) into %fixed-stack.3, align 16)
; MIR32-DAG:  STW killed $r29, -12, $r1 :: (store (s32) into %fixed-stack.2)
; MIR32-DAG:  STW killed $r30, -8, $r1 :: (store (s32) into %fixed-stack.1, align 8)
; MIR32-DAG:  STW killed $r31, -4, $r1 :: (store (s32) into %fixed-stack.0)

; MIR32:      INLINEASM

; MIR32-DAG:  $r31 = LWZ -4, $r1 :: (load (s32) from %fixed-stack.0)
; MIR32-DAG:  $r30 = LWZ -8, $r1 :: (load (s32) from %fixed-stack.1, align 8)
; MIR32-DAG:  $r29 = LWZ -12, $r1 :: (load (s32) from %fixed-stack.2)
; MIR32-DAG:  $r28 = LWZ -16, $r1 :: (load (s32) from %fixed-stack.3, align 16)
; MIR32-DAG:  $r27 = LWZ -20, $r1 :: (load (s32) from %fixed-stack.4)
; MIR32-DAG:  $r26 = LWZ -24, $r1 :: (load (s32) from %fixed-stack.5, align 8)
; MIR32-DAG:  $r25 = LWZ -28, $r1 :: (load (s32) from %fixed-stack.6)
; MIR32-DAG:  $r24 = LWZ -32, $r1 :: (load (s32) from %fixed-stack.7, align 16)
; MIR32-DAG:  $r23 = LWZ -36, $r1 :: (load (s32) from %fixed-stack.8)
; MIR32-DAG:  $r22 = LWZ -40, $r1 :: (load (s32) from %fixed-stack.9, align 8)
; MIR32-DAG:  $r21 = LWZ -44, $r1 :: (load (s32) from %fixed-stack.10)
; MIR32-DAG:  $r20 = LWZ -48, $r1 :: (load (s32) from %fixed-stack.11, align 16)
; MIR32-DAG:  $r19 = LWZ -52, $r1 :: (load (s32) from %fixed-stack.12)
; MIR32-DAG:  $r18 = LWZ -56, $r1 :: (load (s32) from %fixed-stack.13, align 8)
; MIR32-DAG:  $r17 = LWZ -60, $r1 :: (load (s32) from %fixed-stack.14)
; MIR32-DAG:  $r16 = LWZ -64, $r1 :: (load (s32) from %fixed-stack.15, align 16)
; MIR32:      BLR implicit $lr, implicit $rm, implicit $r3


; ASM64-LABEL: .gprs_only:
; ASM64-DAG:     std 16, -128(1)                         # 8-byte Folded Spill
; ASM64-DAG:     std 17, -120(1)                         # 8-byte Folded Spill
; ASM64-DAG:     std 18, -112(1)                         # 8-byte Folded Spill
; ASM64-DAG:     std 19, -104(1)                         # 8-byte Folded Spill
; ASM64-DAG:     std 20, -96(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 21, -88(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 22, -80(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 23, -72(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 24, -64(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 25, -56(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 26, -48(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 27, -40(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 28, -32(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 29, -24(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 30, -16(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 31, -8(1)                           # 8-byte Folded Spill
; ASM64:         #APP
; AMS64-DAG:     ld 31, -8(1)                            # 8-byte Folded Reload
; ASM64-DAG:     ld 30, -16(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 29, -24(1)                           # 8-byte Folded Reload
; ASM64-DAG:      ld 28, -32(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 27, -40(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 26, -48(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 25, -56(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 24, -64(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 23, -72(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 22, -80(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 21, -88(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 20, -96(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 19, -104(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 18, -112(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 17, -120(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 16, -128(1)                          # 8-byte Folded Reload
; ASM64:          blr

; ASM32-LABEL: .gprs_only:
; ASM32-DAG:     stw 16, -64(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 17, -60(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 18, -56(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 19, -52(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 20, -48(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 21, -44(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 22, -40(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 23, -36(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 24, -32(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 25, -28(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 26, -24(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 27, -20(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 28, -16(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 29, -12(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stw 30, -8(1)                           # 4-byte Folded Spill
; ASM32-DAG:     stw 31, -4(1)                           # 4-byte Folded Spill
; ASM32:         #APP
; ASM32-DAG:     lwz 31, -4(1)                           # 4-byte Folded Reload
; ASM32-DAG:     lwz 30, -8(1)                           # 4-byte Folded Reload
; ASM32-DAG:     lwz 29, -12(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 28, -16(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 27, -20(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 26, -24(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 25, -28(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 24, -32(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 23, -36(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 22, -40(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 21, -44(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 20, -48(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 19, -52(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 18, -56(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 17, -60(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 16, -64(1)                          # 4-byte Folded Reload
; ASM32-DAG:     blr


declare double @dummy(i32 signext);

define dso_local double @fprs_and_gprs(i32 signext %i) {
  call void asm sideeffect "", "~{r13},~{r14},~{r25},~{r31},~{f14},~{f19},~{f21},~{f31}"()
  %result = call double @dummy(i32 signext %i)
  ret double %result
}

; MIR64:       name:            fprs_and_gprs
; MIR64-LABEL: fixedStack:
; MIR64-NEXT: - { id: 0, type: spill-slot, offset: -8, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 1, type: spill-slot, offset: -16, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f30', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 2, type: spill-slot, offset: -24, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f29', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 3, type: spill-slot, offset: -32, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f28', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 4, type: spill-slot, offset: -40, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f27', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 5, type: spill-slot, offset: -48, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f26', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 6, type: spill-slot, offset: -56, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f25', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 7, type: spill-slot, offset: -64, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f24', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 8, type: spill-slot, offset: -72, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f23', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 9, type: spill-slot, offset: -80, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f22', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 10, type: spill-slot, offset: -88, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f21', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 11, type: spill-slot, offset: -96, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f20', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 12, type: spill-slot, offset: -104, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f19', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 13, type: spill-slot, offset: -112, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f18', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 14, type: spill-slot, offset: -120, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f17', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 15, type: spill-slot, offset: -128, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f16', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 16, type: spill-slot, offset: -136, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f15', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 17, type: spill-slot, offset: -144, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$f14', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 18, type: spill-slot, offset: -152, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 19, type: spill-slot, offset: -160, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x30', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 20, type: spill-slot, offset: -168, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x29', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 21, type: spill-slot, offset: -176, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x28', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 22, type: spill-slot, offset: -184, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x27', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 23, type: spill-slot, offset: -192, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x26', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 24, type: spill-slot, offset: -200, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x25', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 25, type: spill-slot, offset: -208, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x24', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 26, type: spill-slot, offset: -216, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x23', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 27, type: spill-slot, offset: -224, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x22', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 28, type: spill-slot, offset: -232, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x21', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 29, type: spill-slot, offset: -240, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x20', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 30, type: spill-slot, offset: -248, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x19', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 31, type: spill-slot, offset: -256, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x18', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 32, type: spill-slot, offset: -264, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x17', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 33, type: spill-slot, offset: -272, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x16', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 34, type: spill-slot, offset: -280, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x15', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT: - { id: 35, type: spill-slot, offset: -288, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:     callee-saved-register: '$x14', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:     debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:  stack:           []

; MIR32:       name:            fprs_and_gprs
; MIR32-LABEL: fixedStack:
; MIR32-NEXT:  - { id: 0, type: spill-slot, offset: -8, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 1, type: spill-slot, offset: -16, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f30', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 2, type: spill-slot, offset: -24, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f29', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 3, type: spill-slot, offset: -32, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f28', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 4, type: spill-slot, offset: -40, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f27', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 5, type: spill-slot, offset: -48, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f26', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 6, type: spill-slot, offset: -56, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f25', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 7, type: spill-slot, offset: -64, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f24', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 8, type: spill-slot, offset: -72, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f23', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 9, type: spill-slot, offset: -80, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f22', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 10, type: spill-slot, offset: -88, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f21', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 11, type: spill-slot, offset: -96, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f20', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 12, type: spill-slot, offset: -104, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f19', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 13, type: spill-slot, offset: -112, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f18', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 14, type: spill-slot, offset: -120, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f17', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 15, type: spill-slot, offset: -128, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f16', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 16, type: spill-slot, offset: -136, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f15', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 17, type: spill-slot, offset: -144, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$f14', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 18, type: spill-slot, offset: -148, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 19, type: spill-slot, offset: -152, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r30', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 20, type: spill-slot, offset: -156, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r29', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 21, type: spill-slot, offset: -160, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r28', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 22, type: spill-slot, offset: -164, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r27', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 23, type: spill-slot, offset: -168, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r26', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 24, type: spill-slot, offset: -172, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r25', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 25, type: spill-slot, offset: -176, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r24', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 26, type: spill-slot, offset: -180, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r23', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 27, type: spill-slot, offset: -184, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r22', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 28, type: spill-slot, offset: -188, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r21', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 29, type: spill-slot, offset: -192, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r20', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 30, type: spill-slot, offset: -196, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r19', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 31, type: spill-slot, offset: -200, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r18', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 32, type: spill-slot, offset: -204, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r17', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 33, type: spill-slot, offset: -208, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r16', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 34, type: spill-slot, offset: -212, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r15', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 35, type: spill-slot, offset: -216, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r14', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  - { id: 36, type: spill-slot, offset: -220, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:      callee-saved-register: '$r13', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:      debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:  stack:           []


; MIR64: liveins: $x3, $x14, $x15, $x16, $x17, $x18, $x19, $x20, $x21, $x22, $x23, $x24, $x25, $x26, $x27, $x28, $x29, $x30, $x31, $f14, $f15, $f16, $f17, $f18, $f19, $f20, $f21, $f22, $f23, $f24, $f25, $f26, $f27, $f28, $f29, $f30, $f31

; MIR64:       $x0 = MFLR8 implicit $lr8
; MIR64-NEXT:  $x1 = STDU $x1, -400, $x1
; MIR64-NEXT:  STD killed $x0, 416, $x1
; MIR64-DAG:   STD killed $x14, 112, $x1 :: (store (s64) into %fixed-stack.35, align 16)
; MIR64-DAG:   STD killed $x15, 120, $x1 :: (store (s64) into %fixed-stack.34)
; MIR64-DAG:   STD killed $x16, 128, $x1 :: (store (s64) into %fixed-stack.33, align 16)
; MIR64-DAG:   STD killed $x17, 136, $x1 :: (store (s64) into %fixed-stack.32)
; MIR64-DAG:   STD killed $x18, 144, $x1 :: (store (s64) into %fixed-stack.31, align 16)
; MIR64-DAG:   STD killed $x19, 152, $x1 :: (store (s64) into %fixed-stack.30)
; MIR64-DAG:   STD killed $x20, 160, $x1 :: (store (s64) into %fixed-stack.29, align 16)
; MIR64-DAG:   STD killed $x21, 168, $x1 :: (store (s64) into %fixed-stack.28)
; MIR64-DAG:   STD killed $x22, 176, $x1 :: (store (s64) into %fixed-stack.27, align 16)
; MIR64-DAG:   STD killed $x23, 184, $x1 :: (store (s64) into %fixed-stack.26)
; MIR64-DAG:   STD killed $x24, 192, $x1 :: (store (s64) into %fixed-stack.25, align 16)
; MIR64-DAG:   STD killed $x25, 200, $x1 :: (store (s64) into %fixed-stack.24)
; MIR64-DAG:   STD killed $x26, 208, $x1 :: (store (s64) into %fixed-stack.23, align 16)
; MIR64-DAG:   STD killed $x27, 216, $x1 :: (store (s64) into %fixed-stack.22)
; MIR64-DAG:   STD killed $x28, 224, $x1 :: (store (s64) into %fixed-stack.21, align 16)
; MIR64-DAG:   STD killed $x29, 232, $x1 :: (store (s64) into %fixed-stack.20)
; MIR64-DAG:   STD killed $x30, 240, $x1 :: (store (s64) into %fixed-stack.19, align 16)
; MIR64-DAG:   STD killed $x31, 248, $x1 :: (store (s64) into %fixed-stack.18)
; MIR64-DAG:   STFD killed $f14, 256, $x1 :: (store (s64) into %fixed-stack.17, align 16)
; MIR64-DAG:   STFD killed $f15, 264, $x1 :: (store (s64) into %fixed-stack.16)
; MIR64-DAG:   STFD killed $f16, 272, $x1 :: (store (s64) into %fixed-stack.15, align 16)
; MIR64-DAG:   STFD killed $f17, 280, $x1 :: (store (s64) into %fixed-stack.14)
; MIR64-DAG:   STFD killed $f18, 288, $x1 :: (store (s64) into %fixed-stack.13, align 16)
; MIR64-DAG:   STFD killed $f19, 296, $x1 :: (store (s64) into %fixed-stack.12)
; MIR64-DAG:   STFD killed $f20, 304, $x1 :: (store (s64) into %fixed-stack.11, align 16)
; MIR64-DAG:   STFD killed $f21, 312, $x1 :: (store (s64) into %fixed-stack.10)
; MIR64-DAG:   STFD killed $f22, 320, $x1 :: (store (s64) into %fixed-stack.9, align 16)
; MIR64-DAG:   STFD killed $f23, 328, $x1 :: (store (s64) into %fixed-stack.8)
; MIR64-DAG:   STFD killed $f24, 336, $x1 :: (store (s64) into %fixed-stack.7, align 16)
; MIR64-DAG:   STFD killed $f25, 344, $x1 :: (store (s64) into %fixed-stack.6)
; MIR64-DAG:   STFD killed $f26, 352, $x1 :: (store (s64) into %fixed-stack.5, align 16)
; MIR64-DAG:   STFD killed $f27, 360, $x1 :: (store (s64) into %fixed-stack.4)
; MIR64-DAG:   STFD killed $f28, 368, $x1 :: (store (s64) into %fixed-stack.3, align 16)
; MIR64-DAG:   STFD killed $f29, 376, $x1 :: (store (s64) into %fixed-stack.2)
; MIR64-DAG:   STFD killed $f30, 384, $x1 :: (store (s64) into %fixed-stack.1, align 16)
; MIR64-DAG:   STFD killed $f31, 392, $x1 :: (store (s64) into %fixed-stack.0)

; MIR64:       INLINEASM
; MIR64-NEXT:  BL8_NOP

; MIR64-DAG:   $f31 = LFD 392, $x1 :: (load (s64) from %fixed-stack.0)
; MIR64-DAG:   $f30 = LFD 384, $x1 :: (load (s64) from %fixed-stack.1, align 16)
; MIR64-DAG:   $f29 = LFD 376, $x1 :: (load (s64) from %fixed-stack.2)
; MIR64-DAG:   $f28 = LFD 368, $x1 :: (load (s64) from %fixed-stack.3, align 16)
; MIR64-DAG:   $f27 = LFD 360, $x1 :: (load (s64) from %fixed-stack.4)
; MIR64-DAG:   $f26 = LFD 352, $x1 :: (load (s64) from %fixed-stack.5, align 16)
; MIR64-DAG:   $f25 = LFD 344, $x1 :: (load (s64) from %fixed-stack.6)
; MIR64-DAG:   $f24 = LFD 336, $x1 :: (load (s64) from %fixed-stack.7, align 16)
; MIR64-DAG:   $f23 = LFD 328, $x1 :: (load (s64) from %fixed-stack.8)
; MIR64-DAG:   $f22 = LFD 320, $x1 :: (load (s64) from %fixed-stack.9, align 16)
; MIR64-DAG:   $f21 = LFD 312, $x1 :: (load (s64) from %fixed-stack.10)
; MIR64-DAG:   $f20 = LFD 304, $x1 :: (load (s64) from %fixed-stack.11, align 16)
; MIR64-DAG:   $f19 = LFD 296, $x1 :: (load (s64) from %fixed-stack.12)
; MIR64-DAG:   $f18 = LFD 288, $x1 :: (load (s64) from %fixed-stack.13, align 16)
; MIR64-DAG:   $f17 = LFD 280, $x1 :: (load (s64) from %fixed-stack.14)
; MIR64-DAG:   $f16 = LFD 272, $x1 :: (load (s64) from %fixed-stack.15, align 16)
; MIR64-DAG:   $f15 = LFD 264, $x1 :: (load (s64) from %fixed-stack.16)
; MIR64-DAG:   $f14 = LFD 256, $x1 :: (load (s64) from %fixed-stack.17, align 16)
; MIR64-DAG:   $x31 = LD 248, $x1 :: (load (s64) from %fixed-stack.18)
; MIR64-DAG:   $x30 = LD 240, $x1 :: (load (s64) from %fixed-stack.19, align 16)
; MIR64-DAG:   $x29 = LD 232, $x1 :: (load (s64) from %fixed-stack.20)
; MIR64-DAG:   $x28 = LD 224, $x1 :: (load (s64) from %fixed-stack.21, align 16)
; MIR64-DAG:   $x27 = LD 216, $x1 :: (load (s64) from %fixed-stack.22)
; MIR64-DAG:   $x26 = LD 208, $x1 :: (load (s64) from %fixed-stack.23, align 16)
; MIR64-DAG:   $x25 = LD 200, $x1 :: (load (s64) from %fixed-stack.24)
; MIR64-DAG:   $x24 = LD 192, $x1 :: (load (s64) from %fixed-stack.25, align 16)
; MIR64-DAG:   $x23 = LD 184, $x1 :: (load (s64) from %fixed-stack.26)
; MIR64-DAG:   $x22 = LD 176, $x1 :: (load (s64) from %fixed-stack.27, align 16)
; MIR64-DAG:   $x21 = LD 168, $x1 :: (load (s64) from %fixed-stack.28)
; MIR64-DAG:   $x20 = LD 160, $x1 :: (load (s64) from %fixed-stack.29, align 16)
; MIR64-DAG:   $x19 = LD 152, $x1 :: (load (s64) from %fixed-stack.30)
; MIR64-DAG:   $x18 = LD 144, $x1 :: (load (s64) from %fixed-stack.31, align 16)
; MIR64-DAG:   $x17 = LD 136, $x1 :: (load (s64) from %fixed-stack.32)
; MIR64-DAG:   $x16 = LD 128, $x1 :: (load (s64) from %fixed-stack.33, align 16)
; MIR64-DAG:   $x15 = LD 120, $x1 :: (load (s64) from %fixed-stack.34)
; MIR64-DAG:   $x14 = LD 112, $x1 :: (load (s64) from %fixed-stack.35, align 16)

; MIR64:       $x1 = ADDI8 $x1, 400
; MIR64-NEXT:  $x0 = LD 16, $x1
; MIR64-NEXT:  MTLR8 $x0, implicit-def $lr8
; MIR64-NEXT:  BLR8 implicit $lr8, implicit $rm, implicit $f1

; MIR32: liveins: $r3, $r13, $r14, $r15, $r16, $r17, $r18, $r19, $r20, $r21, $r22, $r23, $r24, $r25, $r26, $r27, $r28, $r29, $r30, $r31, $f14, $f15, $f16, $f17, $f18, $f19, $f20, $f21, $f22, $f23, $f24, $f25, $f26, $f27, $f28, $f29, $f30, $f31

; MIR32:      $r0 = MFLR implicit $lr
; MIR32-NEXT: $r1 = STWU $r1, -288, $r1
; MIR32-NEXT: STW killed $r0, 296, $r1
; MIR32-DAG:  STW killed $r13, 68, $r1 :: (store (s32) into %fixed-stack.36)
; MIR32-DAG:  STW killed $r14, 72, $r1 :: (store (s32) into %fixed-stack.35, align 8)
; MIR32-DAG:  STW killed $r15, 76, $r1 :: (store (s32) into %fixed-stack.34)
; MIR32-DAG:  STW killed $r16, 80, $r1 :: (store (s32) into %fixed-stack.33, align 16)
; MIR32-DAG:  STW killed $r17, 84, $r1 :: (store (s32) into %fixed-stack.32)
; MIR32-DAG:  STW killed $r18, 88, $r1 :: (store (s32) into %fixed-stack.31, align 8)
; MIR32-DAG:  STW killed $r19, 92, $r1 :: (store (s32) into %fixed-stack.30)
; MIR32-DAG:  STW killed $r20, 96, $r1 :: (store (s32) into %fixed-stack.29, align 16)
; MIR32-DAG:  STW killed $r21, 100, $r1 :: (store (s32) into %fixed-stack.28)
; MIR32-DAG:  STW killed $r22, 104, $r1 :: (store (s32) into %fixed-stack.27, align 8)
; MIR32-DAG:  STW killed $r23, 108, $r1 :: (store (s32) into %fixed-stack.26)
; MIR32-DAG:  STW killed $r24, 112, $r1 :: (store (s32) into %fixed-stack.25, align 16)
; MIR32-DAG:  STW killed $r25, 116, $r1 :: (store (s32) into %fixed-stack.24)
; MIR32-DAG:  STW killed $r26, 120, $r1 :: (store (s32) into %fixed-stack.23, align 8)
; MIR32-DAG:  STW killed $r27, 124, $r1 :: (store (s32) into %fixed-stack.22)
; MIR32-DAG:  STW killed $r28, 128, $r1 :: (store (s32) into %fixed-stack.21, align 16)
; MIR32-DAG:  STW killed $r29, 132, $r1 :: (store (s32) into %fixed-stack.20)
; MIR32-DAG:  STW killed $r30, 136, $r1 :: (store (s32) into %fixed-stack.19, align 8)
; MIR32-DAG:  STW killed $r31, 140, $r1 :: (store (s32) into %fixed-stack.18)
; MIR32-DAG:  STFD killed $f14, 144, $r1 :: (store (s64) into %fixed-stack.17, align 16)
; MIR32-DAG:  STFD killed $f15, 152, $r1 :: (store (s64) into %fixed-stack.16)
; MIR32-DAG:  STFD killed $f16, 160, $r1 :: (store (s64) into %fixed-stack.15, align 16)
; MIR32-DAG:  STFD killed $f17, 168, $r1 :: (store (s64) into %fixed-stack.14)
; MIR32-DAG:  STFD killed $f18, 176, $r1 :: (store (s64) into %fixed-stack.13, align 16)
; MIR32-DAG:  STFD killed $f19, 184, $r1 :: (store (s64) into %fixed-stack.12)
; MIR32-DAG:  STFD killed $f20, 192, $r1 :: (store (s64) into %fixed-stack.11, align 16)
; MIR32-DAG:  STFD killed $f21, 200, $r1 :: (store (s64) into %fixed-stack.10)
; MIR32-DAG:  STFD killed $f22, 208, $r1 :: (store (s64) into %fixed-stack.9, align 16)
; MIR32-DAG:  STFD killed $f23, 216, $r1 :: (store (s64) into %fixed-stack.8)
; MIR32-DAG:  STFD killed $f24, 224, $r1 :: (store (s64) into %fixed-stack.7, align 16)
; MIR32-DAG:  STFD killed $f25, 232, $r1 :: (store (s64) into %fixed-stack.6)
; MIR32-DAG:  STFD killed $f26, 240, $r1 :: (store (s64) into %fixed-stack.5, align 16)
; MIR32-DAG:  STFD killed $f27, 248, $r1 :: (store (s64) into %fixed-stack.4)
; MIR32-DAG:  STFD killed $f28, 256, $r1 :: (store (s64) into %fixed-stack.3, align 16)
; MIR32-DAG:  STFD killed $f29, 264, $r1 :: (store (s64) into %fixed-stack.2)
; MIR32-DAG:  STFD killed $f30, 272, $r1 :: (store (s64) into %fixed-stack.1, align 16)
; MIR32-DAG:  STFD killed $f31, 280, $r1 :: (store (s64) into %fixed-stack.0)

; MIR32:      INLINEASM
; MIR32:      BL_NOP

; MIR32-DAG:  $f31 = LFD 280, $r1 :: (load (s64) from %fixed-stack.0)
; MIR32-DAG:  $f30 = LFD 272, $r1 :: (load (s64) from %fixed-stack.1, align 16)
; MIR32-DAG:  $f29 = LFD 264, $r1 :: (load (s64) from %fixed-stack.2)
; MIR32-DAG:  $f28 = LFD 256, $r1 :: (load (s64) from %fixed-stack.3, align 16)
; MIR32-DAG:  $f27 = LFD 248, $r1 :: (load (s64) from %fixed-stack.4)
; MIR32-DAG:  $f26 = LFD 240, $r1 :: (load (s64) from %fixed-stack.5, align 16)
; MIR32-DAG:  $f25 = LFD 232, $r1 :: (load (s64) from %fixed-stack.6)
; MIR32-DAG:  $f24 = LFD 224, $r1 :: (load (s64) from %fixed-stack.7, align 16)
; MIR32-DAG:  $f23 = LFD 216, $r1 :: (load (s64) from %fixed-stack.8)
; MIR32-DAG:  $f22 = LFD 208, $r1 :: (load (s64) from %fixed-stack.9, align 16)
; MIR32-DAG:  $f21 = LFD 200, $r1 :: (load (s64) from %fixed-stack.10)
; MIR32-DAG:  $f20 = LFD 192, $r1 :: (load (s64) from %fixed-stack.11, align 16)
; MIR32-DAG:  $f19 = LFD 184, $r1 :: (load (s64) from %fixed-stack.12)
; MIR32-DAG:  $f18 = LFD 176, $r1 :: (load (s64) from %fixed-stack.13, align 16)
; MIR32-DAG:  $f17 = LFD 168, $r1 :: (load (s64) from %fixed-stack.14)
; MIR32-DAG:  $f16 = LFD 160, $r1 :: (load (s64) from %fixed-stack.15, align 16)
; MIR32-DAG:  $f15 = LFD 152, $r1 :: (load (s64) from %fixed-stack.16)
; MIR32-DAG:  $f14 = LFD 144, $r1 :: (load (s64) from %fixed-stack.17, align 16)
; MIR32-DAG:  $r31 = LWZ 140, $r1 :: (load (s32) from %fixed-stack.18)
; MIR32-DAG:  $r30 = LWZ 136, $r1 :: (load (s32) from %fixed-stack.19, align 8)
; MIR32-DAG:  $r29 = LWZ 132, $r1 :: (load (s32) from %fixed-stack.20)
; MIR32-DAG:  $r28 = LWZ 128, $r1 :: (load (s32) from %fixed-stack.21, align 16)
; MIR32-DAG:  $r27 = LWZ 124, $r1 :: (load (s32) from %fixed-stack.22)
; MIR32-DAG:  $r26 = LWZ 120, $r1 :: (load (s32) from %fixed-stack.23, align 8)
; MIR32-DAG:  $r25 = LWZ 116, $r1 :: (load (s32) from %fixed-stack.24)
; MIR32-DAG:  $r24 = LWZ 112, $r1 :: (load (s32) from %fixed-stack.25, align 16)
; MIR32-DAG:  $r23 = LWZ 108, $r1 :: (load (s32) from %fixed-stack.26)
; MIR32-DAG:  $r22 = LWZ 104, $r1 :: (load (s32) from %fixed-stack.27, align 8)
; MIR32-DAG:  $r21 = LWZ 100, $r1 :: (load (s32) from %fixed-stack.28)
; MIR32-DAG:  $r20 = LWZ 96, $r1 :: (load (s32) from %fixed-stack.29, align 16)
; MIR32-DAG:  $r19 = LWZ 92, $r1 :: (load (s32) from %fixed-stack.30)
; MIR32-DAG:  $r18 = LWZ 88, $r1 :: (load (s32) from %fixed-stack.31, align 8)
; MIR32-DAG:  $r17 = LWZ 84, $r1 :: (load (s32) from %fixed-stack.32)
; MIR32-DAG:  $r16 = LWZ 80, $r1 :: (load (s32) from %fixed-stack.33, align 16)
; MIR32-DAG:  $r15 = LWZ 76, $r1 :: (load (s32) from %fixed-stack.34)
; MIR32-DAG:  $r14 = LWZ 72, $r1 :: (load (s32) from %fixed-stack.35, align 8)
; MIR32-DAG:  $r13 = LWZ 68, $r1 :: (load (s32) from %fixed-stack.36)
; MIR32:      $r1 = ADDI $r1, 288
; MIR32-NEXT: $r0 = LWZ 8, $r1
; MIR32-NEXT: MTLR $r0, implicit-def $lr
; MIR32-NEXT: BLR implicit $lr, implicit $rm, implicit $f1

; ASM64-LABEL: .fprs_and_gprs:
; ASM64:         mflr 0
; ASM64-NEXT:    stdu 1, -400(1)
; ASM64-NEXT:    std 0, 416(1)
; ASM64-DAG:     std 14, 112(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 15, 120(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 16, 128(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 17, 136(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 18, 144(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 19, 152(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 20, 160(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 21, 168(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 22, 176(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 23, 184(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 24, 192(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 25, 200(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 26, 208(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 27, 216(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 28, 224(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 29, 232(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 30, 240(1)                          # 8-byte Folded Spill
; ASM64-DAG:     std 31, 248(1)                          # 8-byte Folded Spill
; ASM64-DAG:     stfd 14, 256(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 15, 264(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 16, 272(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 17, 280(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 18, 288(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 19, 296(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 20, 304(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 21, 312(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 22, 320(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 23, 328(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 24, 336(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 25, 344(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 26, 352(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 27, 360(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 28, 368(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 29, 376(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 30, 384(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 31, 392(1)                         # 8-byte Folded Spill

; ASM64:         bl .dummy
; ASM64-DAG:     lfd 31, 392(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 30, 384(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 29, 376(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 28, 368(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 27, 360(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 26, 352(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 25, 344(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 24, 336(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 23, 328(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 22, 320(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 21, 312(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 20, 304(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 19, 296(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 18, 288(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 17, 280(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 16, 272(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 15, 264(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 14, 256(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 31, 248(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 30, 240(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 29, 232(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 28, 224(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 27, 216(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 26, 208(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 25, 200(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 24, 192(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 23, 184(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 22, 176(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 21, 168(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 20, 160(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 19, 152(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 18, 144(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 17, 136(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 16, 128(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 15, 120(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 14, 112(1)                           # 8-byte Folded Reload

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

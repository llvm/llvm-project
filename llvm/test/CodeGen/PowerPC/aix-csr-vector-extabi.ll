; RUN: llc -mtriple=powerpc-unknown-aix-xcoff -vec-extabi -verify-machineinstrs -mcpu=pwr7 \
; RUN:     -mattr=+altivec -stop-after=prologepilog < %s | \
; RUN:   FileCheck --check-prefix=MIR32 %s

; RUN: llc -mtriple=powerpc-unknown-aix-xcoff -vec-extabi -verify-machineinstrs \
; RUN:     -mcpu=pwr7 -mattr=+altivec < %s | \
; RUN:   FileCheck --check-prefix=ASM32 %s

; RUN: llc -mtriple=powerpc64-unknown-aix-xcoff -vec-extabi -verify-machineinstrs \
; RUN:     -mcpu=pwr7 -mattr=+altivec -stop-after=prologepilog < %s | \
; RUN:   FileCheck --check-prefix=MIR64 %s

; RUN: llc -mtriple=powerpc64-unknown-aix-xcoff -vec-extabi -verify-machineinstrs \
; RUN:     -mcpu=pwr7 -mattr=+altivec < %s | \
; RUN:   FileCheck --check-prefix=ASM64 %s


define dso_local void @vec_regs() {
entry:
  call void asm sideeffect "", "~{v13},~{v20},~{v26},~{v31}"()
  ret void
}

; MIR32:         name:            vec_regs

; MIR32-LABEL:   fixedStack:
; MIR32-NEXT:    - { id: 0, type: spill-slot, offset: -16, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$v31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 1, type: spill-slot, offset: -96, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$v26', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 2, type: spill-slot, offset: -192, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$v20', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    stack:

; MIR32:         liveins: $v20, $v26, $v31

; MIR32-DAG:     STXVD2X killed $v20, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.2)
; MIR32-DAG:     STXVD2X killed $v26, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.1)
; MIR32-DAG:     STXVD2X killed $v31, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.0)

; MIR32:         INLINEASM

; MIR32-DAG:     $v20 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.2)
; MIR32-DAG:     $v26 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.1)
; MIR32-DAG:     $v31 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.0)
; MIR32:         BLR implicit $lr, implicit $rm

; MIR64:         name:            vec_regs

; MIR64-LABEL:   fixedStack:
; MIR64-NEXT:    - { id: 0, type: spill-slot, offset: -16, size: 16, alignment: 16, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$v31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 1, type: spill-slot, offset: -96, size: 16, alignment: 16, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$v26', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 2, type: spill-slot, offset: -192, size: 16, alignment: 16, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$v20', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    stack:

; MIR64:         liveins: $v20, $v26, $v31

; MIR64-DAG:     STXVD2X killed $v20, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.2)
; MIR64-DAG:     STXVD2X killed $v26, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.1)
; MIR64-DAG:     STXVD2X killed $v31, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.0)

; MIR64:         INLINEASM

; MIR64-DAG:     $v20 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.2)
; MIR64-DAG:     $v26 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.1)
; MIR64-DAG:     $v31 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.0)
; MIR64:         BLR8 implicit $lr8, implicit $rm


; ASM32-LABEL:   .vec_regs:

; ASM32:         li {{[0-9]+}}, -192
; ASM32-DAG:     stxvd2x 52, 1, {{[0-9]+}}               # 16-byte Folded Spill
; ASM32-DAG:     li {{[0-9]+}}, -96
; ASM32-DAG:     stxvd2x 58, 1, {{[0-9]+}}               # 16-byte Folded Spill
; ASM32-DAG:     li {{[0-9]+}}, -16
; ASM32-DAG:     stxvd2x 63, 1, {{[0-9]+}}               # 16-byte Folded Spill
; ASM32:         #APP
; ASM32-DAG:     #NO_APP
; ASM32-DAG:     lxvd2x 63, 1, {{[0-9]+}}       # 16-byte Folded Reload
; ASM32-DAG:     li {{[0-9]+}}, -96
; ASM32-DAG:     lxvd2x 58, 1, {{[0-9]+}}       # 16-byte Folded Reload
; ASM32-DAG:     li {{[0-9]+}}, -192
; ASM32-DAG:     lxvd2x 52, 1, {{[0-9]+}}       # 16-byte Folded Reload
; ASM32:         blr

; ASM64-LABEL:   .vec_regs:

; ASM64-DAG:     li {{[0-9]+}}, -192
; ASM64-DAG:     stxvd2x 52, 1, {{[0-9]+}}               # 16-byte Folded Spill
; ASM64-DAG:     li {{[0-9]+}}, -96
; ASM64-DAG:     stxvd2x 58, 1, {{[0-9]+}}               # 16-byte Folded Spill
; ASM64-DAG:     li {{[0-9]+}}, -16
; ASM64-DAG:     stxvd2x {{[0-9]+}}, 1, {{[0-9]+}}      # 16-byte Folded Spill
; ASM64-DAG:     #APP
; ASM64-DAG:     #NO_APP
; ASM64-DAG:     lxvd2x {{[0-9]+}}, 1, {{[0-9]+}}       # 16-byte Folded Reload
; ASM64-DAG:     li {{[0-9]+}}, -96
; ASM64-DAG:     lxvd2x 58, 1, {{[0-9]+}}                # 16-byte Folded Reload
; ASM64-DAG:     li {{[0-9]+}}, -192
; ASM64-DAG:     lxvd2x 52, 1, {{[0-9]+}}                # 16-byte Folded Reload
; ASM64-DAG:     blr

define dso_local void @fprs_gprs_vecregs() {
  call void asm sideeffect "", "~{r14},~{r25},~{r31},~{f14},~{f21},~{f31},~{v20},~{v26},~{v31}"()
  ret void
}

; MIR32:         name:            fprs_gprs_vecregs

; MIR32-LABEL:   fixedStack:
; MIR32-NEXT:    - { id: 0, type: spill-slot, offset: -240, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$v31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 1, type: spill-slot, offset: -320, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$v26', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 2, type: spill-slot, offset: -416, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$v20', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 3, type: spill-slot, offset: -8, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$f31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 4, type: spill-slot, offset: -88, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$f21', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 5, type: spill-slot, offset: -144, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$f14', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 6, type: spill-slot, offset: -148, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$r31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 7, type: spill-slot, offset: -172, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$r25', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    - { id: 8, type: spill-slot, offset: -216, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:        callee-saved-register: '$r14', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    stack:

; MIR32:         liveins: $r14, $r25, $r31, $f14, $f21, $f31, $v20, $v26, $v31

; MIR32-DAG:     STW killed $r14, 232, $r1 :: (store (s32) into %fixed-stack.8, align 8)
; MIR32-DAG:     STW killed $r25, 276, $r1 :: (store (s32) into %fixed-stack.7)
; MIR32-DAG:     STW killed $r31, 300, $r1 :: (store (s32) into %fixed-stack.6)
; MIR32-DAG:     STFD killed $f14, 304, $r1 :: (store (s64) into %fixed-stack.5, align 16)
; MIR32-DAG:     STFD killed $f21, 360, $r1 :: (store (s64) into %fixed-stack.4)
; MIR32-DAG:     STFD killed $f31, 440, $r1 :: (store (s64) into %fixed-stack.3)
; MIR32-DAG:     $r{{[0-9]+}} = LI 32
; MIR32-DAG:     STXVD2X killed $v20, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.2)
; MIR32-DAG:     $r{{[0-9]+}} = LI 128
; MIR32-DAG:     STXVD2X killed $v26, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.1)
; MIR32-DAG:     $r{{[0-9]+}} = LI 208
; MIR32-DAG:     STXVD2X killed $v31, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.0)
; MIR32-DAG:     $r1 = STWU $r1, -448, $r1

; MIR32:         INLINEASM

; MIR32-DAG:     $r14 = LWZ 232, $r1 :: (load (s32) from %fixed-stack.8, align 8)
; MIR32-DAG:     $r25 = LWZ 276, $r1 :: (load (s32) from %fixed-stack.7)
; MIR32-DAG:     $r31 = LWZ 300, $r1 :: (load (s32) from %fixed-stack.6)
; MIR32-DAG:     $f14 = LFD 304, $r1 :: (load (s64) from %fixed-stack.5, align 16)
; MIR32-DAG:     $f21 = LFD 360, $r1 :: (load (s64) from %fixed-stack.4)
; MIR32-DAG:     $f31 = LFD 440, $r1 :: (load (s64) from %fixed-stack.3)
; MIR32-DAG:     $v20 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.2)
; MIR32-DAG:     $r{{[0-9]+}} = LI 32
; MIR32-DAG:     $v26 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.1)
; MIR32-DAG:     $r{{[0-9]+}} = LI 128
; MIR32-DAG:     $v31 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.0)
; MIR32-DAG:     $r{{[0-9]+}} = LI 208
; MIR32-DAG:     $r1 = ADDI $r1, 448
; MIR32-DAG:     BLR implicit $lr, implicit $rm

; MIR64:         name:            fprs_gprs_vecregs

; MIR64-LABEL:   fixedStack:
; MIR64-NEXT:    - { id: 0, type: spill-slot, offset: -304, size: 16, alignment: 16, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$v31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 1, type: spill-slot, offset: -384, size: 16, alignment: 16, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$v26', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 2, type: spill-slot, offset: -480, size: 16, alignment: 16, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$v20', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 3, type: spill-slot, offset: -8, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$f31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 4, type: spill-slot, offset: -88, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$f21', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 5, type: spill-slot, offset: -144, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$f14', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 6, type: spill-slot, offset: -152, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$x31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 7, type: spill-slot, offset: -200, size: 8, alignment: 8, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$x25', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    - { id: 8, type: spill-slot, offset: -288, size: 8, alignment: 16, stack-id: default,
; MIR64-NEXT:        callee-saved-register: '$x14', callee-saved-restored: true, debug-info-variable: '',
; MIR64-NEXT:        debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    stack:

; MIR64:         liveins: $x14, $x25, $x31, $f14, $f21, $f31, $v20, $v26, $v31

; MIR64-DAG:     $x1 = STDU $x1, -544, $x1
; MIR64-DAG:     STD killed $x14, 256, $x1 :: (store (s64) into %fixed-stack.8, align 16)
; MIR64-DAG:     STD killed $x25, 344, $x1 :: (store (s64) into %fixed-stack.7)
; MIR64-DAG:     STD killed $x31, 392, $x1 :: (store (s64) into %fixed-stack.6)
; MIR64-DAG:     STFD killed $f14, 400, $x1 :: (store (s64) into %fixed-stack.5, align 16)
; MIR64-DAG:     STFD killed $f21, 456, $x1 :: (store (s64) into %fixed-stack.4)
; MIR64-DAG:     STFD killed $f31, 536, $x1 :: (store (s64) into %fixed-stack.3)
; MIR64-DAG:     $x{{[0-9]+}} = LI8 64
; MIR64-DAG:     STXVD2X killed $v20, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.2)
; MIR64-DAG:     $x{{[0-9]+}} = LI8 160
; MIR64-DAG:     STXVD2X killed $v26, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.1)
; MIR64-DAG:     $x{{[0-9]+}} = LI8 240
; MIR64-DAG:     STXVD2X killed $v31, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.0)

; MIR64:         INLINEASM

; MIR64-DAG:     $x14 = LD 256, $x1 :: (load (s64) from %fixed-stack.8, align 16)
; MIR64-DAG:     $x25 = LD 344, $x1 :: (load (s64) from %fixed-stack.7)
; MIR64-DAG:     $x31 = LD 392, $x1 :: (load (s64) from %fixed-stack.6)
; MIR64-DAG:     $f14 = LFD 400, $x1 :: (load (s64) from %fixed-stack.5, align 16)
; MIR64-DAG:     $f21 = LFD 456, $x1 :: (load (s64) from %fixed-stack.4)
; MIR64-DAG:     $f31 = LFD 536, $x1 :: (load (s64) from %fixed-stack.3)
; MIR64-DAG:     $v20 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.2)
; MIR64-DAG:     $x{{[0-9]+}} = LI8 64
; MIR64-DAG:     $v26 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.1)
; MIR64-DAG:     $x{{[0-9]+}} = LI8 160
; MIR64-DAG:     $v31 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.0)
; MIR64-DAG:     $x{{[0-9]+}} = LI8 240
; MIR64-DAG:     $x1 = ADDI8 $x1, 544
; MIR64-DAG:     BLR8 implicit $lr8, implicit $rm

; ASM32-LABEL:   .fprs_gprs_vecregs:

; ASM32:         stwu 1, -448(1)
; ASM32-DAG:     li {{[0-9]+}}, 32
; ASM32-DAG:     stw 14, 232(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stfd 14, 304(1)                         # 8-byte Folded Spill
; ASM32-DAG:     stxvd2x 52, 1, {{[0-9]+}}               # 16-byte Folded Spill
; ASM32-DAG:     li {{[0-9]+}}, 128
; ASM32-DAG:     stw 25, 276(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stxvd2x 58, 1, {{[0-9]+}}               # 16-byte Folded Spill
; ASM32-DAG:     li {{[0-9]+}}, 208
; ASM32-DAG:     stw 31, 300(1)                          # 4-byte Folded Spill
; ASM32-DAG:     stfd 21, 360(1)                         # 8-byte Folded Spill
; ASM32-DAG:     stfd 31, 440(1)                         # 8-byte Folded Spill
; ASM32-DAG:     stxvd2x 63, 1, {{[0-9]+}}               # 16-byte Folded Spill
; ASM32-DAG:     #APP
; ASM32-DAG:     #NO_APP
; ASM32-DAG:     lxvd2x 63, 1, {{[0-9]+}}                # 16-byte Folded Reload
; ASM32-DAG:     li {{[0-9]+}}, 128
; ASM32-DAG:     lfd 31, 440(1)                          # 8-byte Folded Reload
; ASM32-DAG:     lxvd2x 58, 1, {{[0-9]+}}                # 16-byte Folded Reload
; ASM32-DAG:     li {{[0-9]+}}, 32
; ASM32-DAG:     lfd 21, 360(1)                          # 8-byte Folded Reload
; ASM32-DAG:     lxvd2x 52, 1, {{[0-9]+}}                # 16-byte Folded Reload
; ASM32-DAG:     lfd 14, 304(1)                          # 8-byte Folded Reload
; ASM32-DAG:     lwz 31, 300(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 25, 276(1)                          # 4-byte Folded Reload
; ASM32-DAG:     lwz 14, 232(1)                          # 4-byte Folded Reload
; ASM32-DAG:     addi 1, 1, 448
; ASM32:         blr

; ASM64-LABEL    .fprs_gprs_vecregs:

; ASM64:         stdu 1, -544(1)
; ASM64-DAG:     li {{[0-9]+}}, 64
; ASM64-DAG:     std 14, 256(1)                          # 8-byte Folded Spill
; ASM64-DAG:     stfd 14, 400(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stxvd2x 52, 1, {{[0-9]+}}               # 16-byte Folded Spill
; ASM64-DAG:     li {{[0-9]+}}, 160
; ASM64-DAG:     std 25, 344(1)                          # 8-byte Folded Spill
; ASM64-DAG:     stxvd2x 58, 1, {{[0-9]+}}               # 16-byte Folded Spill
; ASM64-DAG:     li {{[0-9]+}}, 240
; ASM64-DAG:     std 31, 392(1)                          # 8-byte Folded Spill
; ASM64-DAG:     stfd 21, 456(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 31, 536(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stxvd2x 63, 1, {{[0-9]+}}               # 16-byte Folded Spill
; ASM64-DAG:     #APP
; ASM64-DAG:     #NO_APP
; ASM64-DAG:     lxvd2x 63, 1, {{[0-9]+}}                # 16-byte Folded Reload
; ASM64-DAG:     li {{[0-9]+}}, 160
; ASM64-DAG:     lfd 31, 536(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lxvd2x 58, 1, {{[0-9]+}}                # 16-byte Folded Reload
; ASM64-DAG:     li {{[0-9]+}}, 64
; ASM64-DAG:     lfd 21, 456(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lxvd2x 52, 1, {{[0-9]+}}                # 16-byte Folded Reload
; ASM64-DAG:     lfd 14, 400(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 31, 392(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 25, 344(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 14, 256(1)                           # 8-byte Folded Reload
; ASM64-DAG:     addi 1, 1, 544
; ASM64:         blr

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

; MIR32-LABEL:  fixedStack:
; MIR32-NEXT:     - { id: 0, type: spill-slot, offset: -16, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:         callee-saved-register: '$v31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:         debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:     - { id: 1, type: spill-slot, offset: -32, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:         callee-saved-register: '$v30', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:         debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:     - { id: 2, type: spill-slot, offset: -48, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:         callee-saved-register: '$v29', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:         debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:     - { id: 3, type: spill-slot, offset: -64, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:         callee-saved-register: '$v28', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:         debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:     - { id: 4, type: spill-slot, offset: -80, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:         callee-saved-register: '$v27', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:         debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:     - { id: 5, type: spill-slot, offset: -96, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:         callee-saved-register: '$v26', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:         debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:     - { id: 6, type: spill-slot, offset: -112, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:         callee-saved-register: '$v25', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:         debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:     - { id: 7, type: spill-slot, offset: -128, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:         callee-saved-register: '$v24', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:         debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:     - { id: 8, type: spill-slot, offset: -144, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:         callee-saved-register: '$v23', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:         debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:     - { id: 9, type: spill-slot, offset: -160, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:         callee-saved-register: '$v22', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:         debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:     - { id: 10, type: spill-slot, offset: -176, size: 16, alignment: 16,
; MIR32-NEXT:         stack-id: default, callee-saved-register: '$v21', callee-saved-restored: true,
; MIR32-NEXT:         debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:     - { id: 11, type: spill-slot, offset: -192, size: 16, alignment: 16,
; MIR32-NEXT:         stack-id: default, callee-saved-register: '$v20', callee-saved-restored: true,
; MIR32-NEXT:         debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    stack:

; MIR32: liveins: $v20, $v21, $v22, $v23, $v24, $v25, $v26, $v27, $v28, $v29, $v30, $v31

; MIR32-DAG:     STXVD2X killed $v20, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.11)
; MIR32-DAG:     STXVD2X killed $v21, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.10)
; MIR32-DAG:     STXVD2X killed $v22, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.9)
; MIR32-DAG:     STXVD2X killed $v23, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.8)
; MIR32-DAG:     STXVD2X killed $v24, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.7)
; MIR32-DAG:     STXVD2X killed $v25, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.6)
; MIR32-DAG:     STXVD2X killed $v26, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.5)
; MIR32-DAG:     STXVD2X killed $v27, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.4)
; MIR32-DAG:     STXVD2X killed $v28, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.3)
; MIR32-DAG:     STXVD2X killed $v29, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.2)
; MIR32-DAG:     STXVD2X killed $v30, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.1)
; MIR32-DAG:     STXVD2X killed $v31, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.0)

; MIR32:         INLINEASM

; MIR32-DAG:     $v31 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.0)
; MIR32-DAG:     $v30 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.1)
; MIR32-DAG:     $v29 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.2)
; MIR32-DAG:     $v28 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.3)
; MIR32-DAG:     $v27 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.4)
; MIR32-DAG:     $v26 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.5)
; MIR32-DAG:     $v25 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.6)
; MIR32-DAG:     $v24 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.7)
; MIR32-DAG:     $v23 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.8)
; MIR32-DAG:     $v22 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.9)
; MIR32-DAG:     $v21 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.10)
; MIR32-DAG:     $v20 = LXVD2X $r1, killed $r{{[0-9]+}} :: (load (s128) from %fixed-stack.11)
; MIR32:         BLR implicit $lr, implicit $rm

; MIR64:         name:            vec_regs

; MIR64-LABEL:   fixedStack:
; MIR64-DAG:       - { id: 0, type: spill-slot, offset: -16, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 1, type: spill-slot, offset: -32, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v30', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 2, type: spill-slot, offset: -48, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v29', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 3, type: spill-slot, offset: -64, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v28', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 4, type: spill-slot, offset: -80, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v27', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 5, type: spill-slot, offset: -96, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v26', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 6, type: spill-slot, offset: -112, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v25', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 7, type: spill-slot, offset: -128, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v24', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 8, type: spill-slot, offset: -144, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v23', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 9, type: spill-slot, offset: -160, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v22', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 10, type: spill-slot, offset: -176, size: 16, alignment: 16,
; MIR64-DAG:           stack-id: default, callee-saved-register: '$v21', callee-saved-restored: true,
; MIR64-DAG:           debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 11, type: spill-slot, offset: -192, size: 16, alignment: 16,
; MIR64-DAG:           stack-id: default, callee-saved-register: '$v20', callee-saved-restored: true,
; MIR64-DAG:           debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    stack:

; MIR64: liveins: $v20, $v21, $v22, $v23, $v24, $v25, $v26, $v27, $v28, $v29, $v30, $v31

; MIR64-DAG:   STXVD2X killed $v20, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.11)
; MIR64-DAG:   STXVD2X killed $v21, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.10)
; MIR64-DAG:   STXVD2X killed $v22, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.9)
; MIR64-DAG:   STXVD2X killed $v23, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.8)
; MIR64-DAG:   STXVD2X killed $v24, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.7)
; MIR64-DAG:   STXVD2X killed $v25, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.6)
; MIR64-DAG:   STXVD2X killed $v26, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.5)
; MIR64-DAG:   STXVD2X killed $v27, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.4)
; MIR64-DAG:   STXVD2X killed $v28, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.3)
; MIR64-DAG:   STXVD2X killed $v29, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.2)
; MIR64-DAG:   STXVD2X killed $v30, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.1)
; MIR64-DAG:   STXVD2X killed $v31, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.0)

; MIR64:       INLINEASM

; MIR64-DAG:   $v31 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.0)
; MIR64-DAG:   $v30 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.1)
; MIR64-DAG:   $v29 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.2)
; MIR64-DAG:   $v28 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.3)
; MIR64-DAG:   $v27 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.4)
; MIR64-DAG:   $v26 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.5)
; MIR64-DAG:   $v25 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.6)
; MIR64-DAG:   $v24 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.7)
; MIR64-DAG:   $v23 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.8)
; MIR64-DAG:   $v22 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.9)
; MIR64-DAG:   $v21 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.10)
; MIR64-DAG:   $v20 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.11)
; MIR64:       BLR8 implicit $lr8, implicit $rm


; ASM32-LABEL:   .vec_regs:

; ASM32-DAG:       li [[FIXEDSTACK11:[0-9]+]], -192
; ASM32-DAG:       stxvd2x 52, 1, [[FIXEDSTACK11]]                       # 16-byte Folded Spill
; ASM32-DAG:       li [[FIXEDSTACK10:[0-9]+]], -176
; ASM32-DAG:       stxvd2x 53, 1, [[FIXEDSTACK10]]                       # 16-byte Folded Spill
; ASM32-DAG:       li [[FIXEDSTACK9:[0-9]+]], -160
; ASM32-DAG:       stxvd2x 54, 1, [[FIXEDSTACK9]]                       # 16-byte Folded Spill
; ASM32-DAG:       li [[FIXEDSTACK8:[0-9]+]], -144
; ASM32-DAG:       stxvd2x 55, 1, [[FIXEDSTACK8]]                       # 16-byte Folded Spill
; ASM32-DAG:       li [[FIXEDSTACK7:[0-9]+]], -128
; ASM32-DAG:       stxvd2x 56, 1, [[FIXEDSTACK7]]                       # 16-byte Folded Spill
; ASM32-DAG:       li [[FIXEDSTACK6:[0-9]+]], -112
; ASM32-DAG:       stxvd2x 57, 1, [[FIXEDSTACK6]]                       # 16-byte Folded Spill
; ASM32-DAG:       li [[FIXEDSTACK5:[0-9]+]], -96
; ASM32-DAG:       stxvd2x 58, 1, [[FIXEDSTACK5]]                       # 16-byte Folded Spill
; ASM32-DAG:       li [[FIXEDSTACK4:[0-9]+]], -80
; ASM32-DAG:       stxvd2x 59, 1, [[FIXEDSTACK4]]                       # 16-byte Folded Spill
; ASM32-DAG:       li [[FIXEDSTACK3:[0-9]+]], -64
; ASM32-DAG:       stxvd2x 60, 1, [[FIXEDSTACK3]]                       # 16-byte Folded Spill
; ASM32-DAG:       li [[FIXEDSTACK2:[0-9]+]], -48
; ASM32-DAG:       stxvd2x 61, 1, [[FIXEDSTACK2]]                       # 16-byte Folded Spill
; ASM32-DAG:       li [[FIXEDSTACK1:[0-9]+]], -32
; ASM32-DAG:       stxvd2x 62, 1, [[FIXEDSTACK1]]                       # 16-byte Folded Spill
; ASM32-DAG:       li [[FIXEDSTACK0:[0-9]+]], -16
; ASM32-DAG:       stxvd2x 63, 1, [[FIXEDSTACK0]]                       # 16-byte Folded Spill

; ASM32:           #APP
; ASM32-NEXT:      #NO_APP

; ASM32-DAG:       lxvd2x 63, 1, [[FIXEDSTACK0]]                        # 16-byte Folded Reload
; ASM32-DAG:       li [[FIXEDSTACK1:[0-9]+]], -32
; ASM32-DAG:       lxvd2x 62, 1, [[FIXEDSTACK1]]                        # 16-byte Folded Reload
; ASM32-DAG:       li [[FIXEDSTACK2:[0-9]+]], -48
; ASM32-DAG:       lxvd2x 61, 1, [[FIXEDSTACK2]]                        # 16-byte Folded Reload
; ASM32-DAG:       li [[FIXEDSTACK3:[0-9]+]], -64
; ASM32-DAG:       lxvd2x 60, 1, [[FIXEDSTACK3]]                        # 16-byte Folded Reload
; ASM32-DAG:       li [[FIXEDSTACK4:[0-9]+]], -80
; ASM32-DAG:       lxvd2x 59, 1, [[FIXEDSTACK4]]                        # 16-byte Folded Reload
; ASM32-DAG:       li [[FIXEDSTACK5:[0-9]+]], -96
; ASM32-DAG:       lxvd2x 58, 1, [[FIXEDSTACK5]]                        # 16-byte Folded Reload
; ASM32-DAG:       li [[FIXEDSTACK6:[0-9]+]], -112
; ASM32-DAG:       lxvd2x 57, 1, [[FIXEDSTACK6]]                        # 16-byte Folded Reload
; ASM32-DAG:       li [[FIXEDSTACK7:[0-9]+]], -128
; ASM32-DAG:       lxvd2x 56, 1, [[FIXEDSTACK7]]                        # 16-byte Folded Reload
; ASM32-DAG:       li [[FIXEDSTACK8:[0-9]+]], -144
; ASM32-DAG:       lxvd2x 55, 1, [[FIXEDSTACK8]]                        # 16-byte Folded Reload
; ASM32-DAG:       li [[FIXEDSTACK9:[0-9]+]], -160
; ASM32-DAG:       lxvd2x 54, 1, [[FIXEDSTACK9]]                        # 16-byte Folded Reload
; ASM32-DAG:       li [[FIXEDSTACK10:[0-9]+]], -176
; ASM32-DAG:       lxvd2x 53, 1, [[FIXEDSTACK10]]                       # 16-byte Folded Reload
; ASM32-DAG:       li [[FIXEDSTACK11:[0-9]+]], -192
; ASM32-DAG:       lxvd2x 52, 1, [[FIXEDSTACK11]]                       # 16-byte Folded Reload
; ASM32:           blr

; ASM64-LABEL:   .vec_regs:

; ASM64-DAG:       li [[FIXEDSTACK11:[0-9]+]], -192
; ASM64-DAG:       stxvd2x 52, 1, [[FIXEDSTACK11]]                   # 16-byte Folded Spill
; ASM64-DAG:       li [[FIXEDSTACK10:[0-9]+]], -176
; ASM64-DAG:       stxvd2x 53, 1, [[FIXEDSTACK10]]                   # 16-byte Folded Spill
; ASM64-DAG:       li [[FIXEDSTACK9:[0-9]+]], -160
; ASM64-DAG:       stxvd2x 54, 1, [[FIXEDSTACK9]]                    # 16-byte Folded Spill
; ASM64-DAG:       li [[FIXEDSTACK8:[0-9]+]], -144
; ASM64-DAG:       stxvd2x 55, 1, [[FIXEDSTACK8]]                    # 16-byte Folded Spill
; ASM64-DAG:       li [[FIXEDSTACK7:[0-9]+]], -128
; ASM64-DAG:       stxvd2x 56, 1, [[FIXEDSTACK7]]                    # 16-byte Folded Spill
; ASM64-DAG:       li [[FIXEDSTACK6:[0-9]+]], -112
; ASM64-DAG:       stxvd2x 57, 1, [[FIXEDSTACK6]]                    # 16-byte Folded Spill
; ASM64-DAG:       li [[FIXEDSTACK5:[0-9]+]], -96
; ASM64-DAG:       stxvd2x 58, 1, [[FIXEDSTACK5]]                    # 16-byte Folded Spill
; ASM64-DAG:       li [[FIXEDSTACK4:[0-9]+]], -80
; ASM64-DAG:       stxvd2x 59, 1, [[FIXEDSTACK4]]                    # 16-byte Folded Spill
; ASM64-DAG:       li [[FIXEDSTACK3:[0-9]+]], -64
; ASM64-DAG:       stxvd2x 60, 1, [[FIXEDSTACK3]]                    # 16-byte Folded Spill
; ASM64-DAG:       li [[FIXEDSTACK2:[0-9]+]], -48
; ASM64-DAG:       stxvd2x 61, 1, [[FIXEDSTACK2]]                    # 16-byte Folded Spill
; ASM64-DAG:       li [[FIXEDSTACK1:[0-9]+]], -32
; ASM64-DAG:       stxvd2x 62, 1, [[FIXEDSTACK1]]                    # 16-byte Folded Spill
; ASM64-DAG:       li [[FIXEDSTACK0:[0-9]+]], -16
; ASM64-DAG:       stxvd2x 63, 1, [[FIXEDSTACK0]]                    # 16-byte Folded Spill

; ASM64-DAG:     #APP
; ASM64-DAG:     #NO_APP

; ASM64-DAG:     lxvd2x 63, 1, [[FIXEDSTACK0]]                         # 16-byte Folded Reload
; ASM64-DAG:     li [[FIXEDSTACK1:[0-9]+]], -32
; ASM64-DAG:     lxvd2x 62, 1, [[FIXEDSTACK1]]                         # 16-byte Folded Reload
; ASM64-DAG:     li [[FIXEDSTACK2:[0-9]+]], -48
; ASM64-DAG:     lxvd2x 61, 1, [[FIXEDSTACK2]]                         # 16-byte Folded Reload
; ASM64-DAG:     li [[FIXEDSTACK3:[0-9]+]], -64
; ASM64-DAG:     lxvd2x 60, 1, [[FIXEDSTACK3]]                         # 16-byte Folded Reload
; ASM64-DAG:     li [[FIXEDSTACK4:[0-9]+]], -80
; ASM64-DAG:     lxvd2x 59, 1, [[FIXEDSTACK4]]                         # 16-byte Folded Reload
; ASM64-DAG:     li [[FIXEDSTACK5:[0-9]+]], -96
; ASM64-DAG:     lxvd2x 58, 1, [[FIXEDSTACK5]]                         # 16-byte Folded Reload
; ASM64-DAG:     li [[FIXEDSTACK6:[0-9]+]], -112
; ASM64-DAG:     lxvd2x 57, 1, [[FIXEDSTACK6]]                         # 16-byte Folded Reload
; ASM64-DAG:     li [[FIXEDSTACK7:[0-9]+]], -128
; ASM64-DAG:     lxvd2x 56, 1, [[FIXEDSTACK7]]                         # 16-byte Folded Reload
; ASM64-DAG:     li [[FIXEDSTACK8:[0-9]+]], -144
; ASM64-DAG:     lxvd2x 55, 1, [[FIXEDSTACK8]]                         # 16-byte Folded Reload
; ASM64-DAG:     li [[FIXEDSTACK9:[0-9]+]], -160
; ASM64-DAG:     lxvd2x 54, 1, [[FIXEDSTACK9]]                         # 16-byte Folded Reload
; ASM64-DAG:     li [[FIXEDSTACK10:[0-9]+]], -176
; ASM64-DAG:     lxvd2x 53, 1, [[FIXEDSTACK10]]                        # 16-byte Folded Reload
; ASM64-DAG:     li [[FIXEDSTACK11:[0-9]+]], -192
; ASM64-DAG:     lxvd2x 52, 1, [[FIXEDSTACK11]]                        # 16-byte Folded Reload

; ASM64:         blr

define dso_local void @fprs_gprs_vecregs() {
  call void asm sideeffect "", "~{r14},~{r25},~{r31},~{f14},~{f21},~{f31},~{v20},~{v26},~{v31}"()
  ret void
}

; MIR32:         name:            fprs_gprs_vecregs

; MIR32-LABEL:   fixedStack:
; MIR32-NEXT:      - { id: 0, type: spill-slot, offset: -240, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$v31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 1, type: spill-slot, offset: -256, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$v30', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 2, type: spill-slot, offset: -272, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$v29', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 3, type: spill-slot, offset: -288, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$v28', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 4, type: spill-slot, offset: -304, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$v27', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 5, type: spill-slot, offset: -320, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$v26', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 6, type: spill-slot, offset: -336, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$v25', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 7, type: spill-slot, offset: -352, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$v24', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 8, type: spill-slot, offset: -368, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$v23', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 9, type: spill-slot, offset: -384, size: 16, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$v22', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 10, type: spill-slot, offset: -400, size: 16, alignment: 16,
; MIR32-NEXT:          stack-id: default, callee-saved-register: '$v21', callee-saved-restored: true,
; MIR32-NEXT:          debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 11, type: spill-slot, offset: -416, size: 16, alignment: 16,
; MIR32-NEXT:          stack-id: default, callee-saved-register: '$v20', callee-saved-restored: true,
; MIR32-NEXT:          debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 12, type: spill-slot, offset: -8, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 13, type: spill-slot, offset: -16, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f30', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 14, type: spill-slot, offset: -24, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f29', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 15, type: spill-slot, offset: -32, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f28', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 16, type: spill-slot, offset: -40, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f27', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 17, type: spill-slot, offset: -48, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f26', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 18, type: spill-slot, offset: -56, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f25', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 19, type: spill-slot, offset: -64, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f24', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 20, type: spill-slot, offset: -72, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f23', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 21, type: spill-slot, offset: -80, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f22', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 22, type: spill-slot, offset: -88, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f21', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 23, type: spill-slot, offset: -96, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f20', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 24, type: spill-slot, offset: -104, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f19', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 25, type: spill-slot, offset: -112, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f18', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 26, type: spill-slot, offset: -120, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f17', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 27, type: spill-slot, offset: -128, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f16', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 28, type: spill-slot, offset: -136, size: 8, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f15', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 29, type: spill-slot, offset: -144, size: 8, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$f14', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 30, type: spill-slot, offset: -148, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r31', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 31, type: spill-slot, offset: -152, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r30', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 32, type: spill-slot, offset: -156, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r29', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 33, type: spill-slot, offset: -160, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r28', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 34, type: spill-slot, offset: -164, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r27', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 35, type: spill-slot, offset: -168, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r26', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 36, type: spill-slot, offset: -172, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r25', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 37, type: spill-slot, offset: -176, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r24', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 38, type: spill-slot, offset: -180, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r23', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 39, type: spill-slot, offset: -184, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r22', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 40, type: spill-slot, offset: -188, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r21', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 41, type: spill-slot, offset: -192, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r20', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 42, type: spill-slot, offset: -196, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r19', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 43, type: spill-slot, offset: -200, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r18', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 44, type: spill-slot, offset: -204, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r17', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 45, type: spill-slot, offset: -208, size: 4, alignment: 16, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r16', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 46, type: spill-slot, offset: -212, size: 4, alignment: 4, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r15', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:      - { id: 47, type: spill-slot, offset: -216, size: 4, alignment: 8, stack-id: default,
; MIR32-NEXT:          callee-saved-register: '$r14', callee-saved-restored: true, debug-info-variable: '',
; MIR32-NEXT:          debug-info-expression: '', debug-info-location: '' }
; MIR32-NEXT:    stack:

; MIR32: liveins: $r14, $r15, $r16, $r17, $r18, $r19, $r20, $r21, $r22, $r23, $r24, $r25, $r26, $r27, $r28, $r29, $r30, $r31, $f14, $f15, $f16, $f17, $f18, $f19, $f20, $f21, $f22, $f23, $f24, $f25, $f26, $f27, $f28, $f29, $f30, $f31, $v20, $v21, $v22, $v23, $v24, $v25, $v26, $v27, $v28, $v29, $v30, $v31

; MIR32-DAG:     STW killed $r14, 232, $r1 :: (store (s32) into %fixed-stack.47, align 8)
; MIR32-DAG:     STW killed $r15, 236, $r1 :: (store (s32) into %fixed-stack.46)
; MIR32-DAG:     STW killed $r16, 240, $r1 :: (store (s32) into %fixed-stack.45, align 16)
; MIR32-DAG:     STW killed $r17, 244, $r1 :: (store (s32) into %fixed-stack.44)
; MIR32-DAG:     STW killed $r18, 248, $r1 :: (store (s32) into %fixed-stack.43, align 8)
; MIR32-DAG:     STW killed $r19, 252, $r1 :: (store (s32) into %fixed-stack.42)
; MIR32-DAG:     STW killed $r20, 256, $r1 :: (store (s32) into %fixed-stack.41, align 16)
; MIR32-DAG:     STW killed $r21, 260, $r1 :: (store (s32) into %fixed-stack.40)
; MIR32-DAG:     STW killed $r22, 264, $r1 :: (store (s32) into %fixed-stack.39, align 8)
; MIR32-DAG:     STW killed $r23, 268, $r1 :: (store (s32) into %fixed-stack.38)
; MIR32-DAG:     STW killed $r24, 272, $r1 :: (store (s32) into %fixed-stack.37, align 16)
; MIR32-DAG:     STW killed $r25, 276, $r1 :: (store (s32) into %fixed-stack.36)
; MIR32-DAG:     STW killed $r26, 280, $r1 :: (store (s32) into %fixed-stack.35, align 8)
; MIR32-DAG:     STW killed $r27, 284, $r1 :: (store (s32) into %fixed-stack.34)
; MIR32-DAG:     STW killed $r28, 288, $r1 :: (store (s32) into %fixed-stack.33, align 16)
; MIR32-DAG:     STW killed $r29, 292, $r1 :: (store (s32) into %fixed-stack.32)
; MIR32-DAG:     STW killed $r30, 296, $r1 :: (store (s32) into %fixed-stack.31, align 8)
; MIR32-DAG:     STW killed $r31, 300, $r1 :: (store (s32) into %fixed-stack.30)
; MIR32-DAG:     STFD killed $f14, 304, $r1 :: (store (s64) into %fixed-stack.29, align 16)
; MIR32-DAG:     STFD killed $f15, 312, $r1 :: (store (s64) into %fixed-stack.28)
; MIR32-DAG:     STFD killed $f16, 320, $r1 :: (store (s64) into %fixed-stack.27, align 16)
; MIR32-DAG:     STFD killed $f17, 328, $r1 :: (store (s64) into %fixed-stack.26)
; MIR32-DAG:     STFD killed $f18, 336, $r1 :: (store (s64) into %fixed-stack.25, align 16)
; MIR32-DAG:     STFD killed $f19, 344, $r1 :: (store (s64) into %fixed-stack.24)
; MIR32-DAG:     STFD killed $f20, 352, $r1 :: (store (s64) into %fixed-stack.23, align 16)
; MIR32-DAG:     STFD killed $f21, 360, $r1 :: (store (s64) into %fixed-stack.22)
; MIR32-DAG:     STFD killed $f22, 368, $r1 :: (store (s64) into %fixed-stack.21, align 16)
; MIR32-DAG:     STFD killed $f23, 376, $r1 :: (store (s64) into %fixed-stack.20)
; MIR32-DAG:     STFD killed $f24, 384, $r1 :: (store (s64) into %fixed-stack.19, align 16)
; MIR32-DAG:     STFD killed $f25, 392, $r1 :: (store (s64) into %fixed-stack.18)
; MIR32-DAG:     STFD killed $f26, 400, $r1 :: (store (s64) into %fixed-stack.17, align 16)
; MIR32-DAG:     STFD killed $f27, 408, $r1 :: (store (s64) into %fixed-stack.16)
; MIR32-DAG:     STFD killed $f28, 416, $r1 :: (store (s64) into %fixed-stack.15, align 16)
; MIR32-DAG:     STFD killed $f29, 424, $r1 :: (store (s64) into %fixed-stack.14)
; MIR32-DAG:     STFD killed $f30, 432, $r1 :: (store (s64) into %fixed-stack.13, align 16)
; MIR32-DAG:     STFD killed $f31, 440, $r1 :: (store (s64) into %fixed-stack.12)
; MIR32-DAG:     STXVD2X killed $v20, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.11)
; MIR32-DAG:     STXVD2X killed $v21, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.10)
; MIR32-DAG:     STXVD2X killed $v22, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.9)
; MIR32-DAG:     STXVD2X killed $v23, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.8)
; MIR32-DAG:     STXVD2X killed $v24, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.7)
; MIR32-DAG:     STXVD2X killed $v25, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.6)
; MIR32-DAG:     STXVD2X killed $v26, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.5)
; MIR32-DAG:     STXVD2X killed $v27, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.4)
; MIR32-DAG:     STXVD2X killed $v28, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.3)
; MIR32-DAG:     STXVD2X killed $v29, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.2)
; MIR32-DAG:     STXVD2X killed $v30, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.1)
; MIR32-DAG:     STXVD2X killed $v31, $r1, killed $r{{[0-9]+}} :: (store (s128) into %fixed-stack.0)

; MIR32:         INLINEASM

; MIR32-DAG:     $v31 = LXVD2X $r1, killed $r3 :: (load (s128) from %fixed-stack.0)
; MIR32-DAG:     $v30 = LXVD2X $r1, killed $r3 :: (load (s128) from %fixed-stack.1)
; MIR32-DAG:     $v29 = LXVD2X $r1, killed $r3 :: (load (s128) from %fixed-stack.2)
; MIR32-DAG:     $v28 = LXVD2X $r1, killed $r3 :: (load (s128) from %fixed-stack.3)
; MIR32-DAG:     $v27 = LXVD2X $r1, killed $r3 :: (load (s128) from %fixed-stack.4)
; MIR32-DAG:     $v26 = LXVD2X $r1, killed $r3 :: (load (s128) from %fixed-stack.5)
; MIR32-DAG:     $v25 = LXVD2X $r1, killed $r3 :: (load (s128) from %fixed-stack.6)
; MIR32-DAG:     $v24 = LXVD2X $r1, killed $r3 :: (load (s128) from %fixed-stack.7)
; MIR32-DAG:     $v23 = LXVD2X $r1, killed $r3 :: (load (s128) from %fixed-stack.8)
; MIR32-DAG:     $v22 = LXVD2X $r1, killed $r3 :: (load (s128) from %fixed-stack.9)
; MIR32-DAG:     $v21 = LXVD2X $r1, killed $r3 :: (load (s128) from %fixed-stack.10)
; MIR32-DAG:     $v20 = LXVD2X $r1, killed $r3 :: (load (s128) from %fixed-stack.11)
; MIR32-DAG:     $f31 = LFD 440, $r1 :: (load (s64) from %fixed-stack.12)
; MIR32-DAG:     $f30 = LFD 432, $r1 :: (load (s64) from %fixed-stack.13, align 16)
; MIR32-DAG:     $f29 = LFD 424, $r1 :: (load (s64) from %fixed-stack.14)
; MIR32-DAG:     $f28 = LFD 416, $r1 :: (load (s64) from %fixed-stack.15, align 16)
; MIR32-DAG:     $f27 = LFD 408, $r1 :: (load (s64) from %fixed-stack.16)
; MIR32-DAG:     $f26 = LFD 400, $r1 :: (load (s64) from %fixed-stack.17, align 16)
; MIR32-DAG:     $f25 = LFD 392, $r1 :: (load (s64) from %fixed-stack.18)
; MIR32-DAG:     $f24 = LFD 384, $r1 :: (load (s64) from %fixed-stack.19, align 16)
; MIR32-DAG:     $f23 = LFD 376, $r1 :: (load (s64) from %fixed-stack.20)
; MIR32-DAG:     $f22 = LFD 368, $r1 :: (load (s64) from %fixed-stack.21, align 16)
; MIR32-DAG:     $f21 = LFD 360, $r1 :: (load (s64) from %fixed-stack.22)
; MIR32-DAG:     $f20 = LFD 352, $r1 :: (load (s64) from %fixed-stack.23, align 16)
; MIR32-DAG:     $f19 = LFD 344, $r1 :: (load (s64) from %fixed-stack.24)
; MIR32-DAG:     $f18 = LFD 336, $r1 :: (load (s64) from %fixed-stack.25, align 16)
; MIR32-DAG:     $f17 = LFD 328, $r1 :: (load (s64) from %fixed-stack.26)
; MIR32-DAG:     $f16 = LFD 320, $r1 :: (load (s64) from %fixed-stack.27, align 16)
; MIR32-DAG:     $f15 = LFD 312, $r1 :: (load (s64) from %fixed-stack.28)
; MIR32-DAG:     $f14 = LFD 304, $r1 :: (load (s64) from %fixed-stack.29, align 16)
; MIR32-DAG:     $r31 = LWZ 300, $r1 :: (load (s32) from %fixed-stack.30)
; MIR32-DAG:     $r30 = LWZ 296, $r1 :: (load (s32) from %fixed-stack.31, align 8)
; MIR32-DAG:     $r29 = LWZ 292, $r1 :: (load (s32) from %fixed-stack.32)
; MIR32-DAG:     $r28 = LWZ 288, $r1 :: (load (s32) from %fixed-stack.33, align 16)
; MIR32-DAG:     $r27 = LWZ 284, $r1 :: (load (s32) from %fixed-stack.34)
; MIR32-DAG:     $r26 = LWZ 280, $r1 :: (load (s32) from %fixed-stack.35, align 8)
; MIR32-DAG:     $r25 = LWZ 276, $r1 :: (load (s32) from %fixed-stack.36)
; MIR32-DAG:     $r24 = LWZ 272, $r1 :: (load (s32) from %fixed-stack.37, align 16)
; MIR32-DAG:     $r23 = LWZ 268, $r1 :: (load (s32) from %fixed-stack.38)
; MIR32-DAG:     $r22 = LWZ 264, $r1 :: (load (s32) from %fixed-stack.39, align 8)
; MIR32-DAG:     $r21 = LWZ 260, $r1 :: (load (s32) from %fixed-stack.40)
; MIR32-DAG:     $r20 = LWZ 256, $r1 :: (load (s32) from %fixed-stack.41, align 16)
; MIR32-DAG:     $r19 = LWZ 252, $r1 :: (load (s32) from %fixed-stack.42)
; MIR32-DAG:     $r18 = LWZ 248, $r1 :: (load (s32) from %fixed-stack.43, align 8)
; MIR32-DAG:     $r17 = LWZ 244, $r1 :: (load (s32) from %fixed-stack.44)
; MIR32-DAG:     $r16 = LWZ 240, $r1 :: (load (s32) from %fixed-stack.45, align 16)
; MIR32-DAG:     $r15 = LWZ 236, $r1 :: (load (s32) from %fixed-stack.46)
; MIR32-DAG:     $r14 = LWZ 232, $r1 :: (load (s32) from %fixed-stack.47, align 8)
; MIR32:         $r1 = ADDI $r1, 448
; MIR32-NEXT:    BLR implicit $lr, implicit $rm


; MIR64:         name:            fprs_gprs_vecregs

; MIR64-LABEL:   fixedStack:
; MIR64-DAG:       - { id: 0, type: spill-slot, offset: -304, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 1, type: spill-slot, offset: -320, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v30', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 2, type: spill-slot, offset: -336, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v29', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 3, type: spill-slot, offset: -352, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v28', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 4, type: spill-slot, offset: -368, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v27', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 5, type: spill-slot, offset: -384, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v26', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 6, type: spill-slot, offset: -400, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v25', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 7, type: spill-slot, offset: -416, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v24', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 8, type: spill-slot, offset: -432, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v23', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 9, type: spill-slot, offset: -448, size: 16, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$v22', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 10, type: spill-slot, offset: -464, size: 16, alignment: 16,
; MIR64-DAG:           stack-id: default, callee-saved-register: '$v21', callee-saved-restored: true,
; MIR64-DAG:           debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 11, type: spill-slot, offset: -480, size: 16, alignment: 16,
; MIR64-DAG:           stack-id: default, callee-saved-register: '$v20', callee-saved-restored: true,
; MIR64-DAG:           debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 12, type: spill-slot, offset: -8, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 13, type: spill-slot, offset: -16, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f30', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 14, type: spill-slot, offset: -24, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f29', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 15, type: spill-slot, offset: -32, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f28', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 16, type: spill-slot, offset: -40, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f27', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 17, type: spill-slot, offset: -48, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f26', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 18, type: spill-slot, offset: -56, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f25', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 19, type: spill-slot, offset: -64, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f24', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 20, type: spill-slot, offset: -72, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f23', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 21, type: spill-slot, offset: -80, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f22', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 22, type: spill-slot, offset: -88, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f21', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 23, type: spill-slot, offset: -96, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f20', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 24, type: spill-slot, offset: -104, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f19', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 25, type: spill-slot, offset: -112, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f18', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 26, type: spill-slot, offset: -120, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f17', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 27, type: spill-slot, offset: -128, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f16', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 28, type: spill-slot, offset: -136, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f15', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 29, type: spill-slot, offset: -144, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$f14', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 30, type: spill-slot, offset: -152, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x31', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 31, type: spill-slot, offset: -160, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x30', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 32, type: spill-slot, offset: -168, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x29', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 33, type: spill-slot, offset: -176, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x28', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 34, type: spill-slot, offset: -184, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x27', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 35, type: spill-slot, offset: -192, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x26', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 36, type: spill-slot, offset: -200, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x25', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 37, type: spill-slot, offset: -208, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x24', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 38, type: spill-slot, offset: -216, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x23', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 39, type: spill-slot, offset: -224, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x22', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 40, type: spill-slot, offset: -232, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x21', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 41, type: spill-slot, offset: -240, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x20', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 42, type: spill-slot, offset: -248, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x19', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 43, type: spill-slot, offset: -256, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x18', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 44, type: spill-slot, offset: -264, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x17', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 45, type: spill-slot, offset: -272, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x16', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 46, type: spill-slot, offset: -280, size: 8, alignment: 8, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x15', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-DAG:       - { id: 47, type: spill-slot, offset: -288, size: 8, alignment: 16, stack-id: default,
; MIR64-DAG:           callee-saved-register: '$x14', callee-saved-restored: true, debug-info-variable: '',
; MIR64-DAG:           debug-info-expression: '', debug-info-location: '' }
; MIR64-NEXT:    stack:

; MIR64: liveins: $x14, $x15, $x16, $x17, $x18, $x19, $x20, $x21, $x22, $x23, $x24, $x25, $x26, $x27, $x28, $x29, $x30, $x31, $f14, $f15, $f16, $f17, $f18, $f19, $f20, $f21, $f22, $f23, $f24, $f25, $f26, $f27, $f28, $f29, $f30, $f31, $v20, $v21, $v22, $v23, $v24, $v25, $v26, $v27, $v28, $v29, $v30, $v31

; MIR64:         $x1 = STDU $x1, -544, $x1
;MIR64-DAG:      STD killed $x14, 256, $x1 :: (store (s64) into %fixed-stack.47, align 16)
;MIR64-DAG:      STD killed $x15, 264, $x1 :: (store (s64) into %fixed-stack.46)
;MIR64-DAG:      STD killed $x16, 272, $x1 :: (store (s64) into %fixed-stack.45, align 16)
;MIR64-DAG:      STD killed $x17, 280, $x1 :: (store (s64) into %fixed-stack.44)
;MIR64-DAG:      STD killed $x18, 288, $x1 :: (store (s64) into %fixed-stack.43, align 16)
;MIR64-DAG:      STD killed $x19, 296, $x1 :: (store (s64) into %fixed-stack.42)
;MIR64-DAG:      STD killed $x20, 304, $x1 :: (store (s64) into %fixed-stack.41, align 16)
;MIR64-DAG:      STD killed $x21, 312, $x1 :: (store (s64) into %fixed-stack.40)
;MIR64-DAG:      STD killed $x22, 320, $x1 :: (store (s64) into %fixed-stack.39, align 16)
;MIR64-DAG:      STD killed $x23, 328, $x1 :: (store (s64) into %fixed-stack.38)
;MIR64-DAG:      STD killed $x24, 336, $x1 :: (store (s64) into %fixed-stack.37, align 16)
;MIR64-DAG:      STD killed $x25, 344, $x1 :: (store (s64) into %fixed-stack.36)
;MIR64-DAG:      STD killed $x26, 352, $x1 :: (store (s64) into %fixed-stack.35, align 16)
;MIR64-DAG:      STD killed $x27, 360, $x1 :: (store (s64) into %fixed-stack.34)
;MIR64-DAG:      STD killed $x28, 368, $x1 :: (store (s64) into %fixed-stack.33, align 16)
;MIR64-DAG:      STD killed $x29, 376, $x1 :: (store (s64) into %fixed-stack.32)
;MIR64-DAG:      STD killed $x30, 384, $x1 :: (store (s64) into %fixed-stack.31, align 16)
;MIR64-DAG:      STD killed $x31, 392, $x1 :: (store (s64) into %fixed-stack.30)
;MIR64-DAG:      STFD killed $f14, 400, $x1 :: (store (s64) into %fixed-stack.29, align 16)
;MIR64-DAG:      STFD killed $f15, 408, $x1 :: (store (s64) into %fixed-stack.28)
;MIR64-DAG:      STFD killed $f16, 416, $x1 :: (store (s64) into %fixed-stack.27, align 16)
;MIR64-DAG:      STFD killed $f17, 424, $x1 :: (store (s64) into %fixed-stack.26)
;MIR64-DAG:      STFD killed $f18, 432, $x1 :: (store (s64) into %fixed-stack.25, align 16)
;MIR64-DAG:      STFD killed $f19, 440, $x1 :: (store (s64) into %fixed-stack.24)
;MIR64-DAG:      STFD killed $f20, 448, $x1 :: (store (s64) into %fixed-stack.23, align 16)
;MIR64-DAG:      STFD killed $f21, 456, $x1 :: (store (s64) into %fixed-stack.22)
;MIR64-DAG:      STFD killed $f22, 464, $x1 :: (store (s64) into %fixed-stack.21, align 16)
;MIR64-DAG:      STFD killed $f23, 472, $x1 :: (store (s64) into %fixed-stack.20)
;MIR64-DAG:      STFD killed $f24, 480, $x1 :: (store (s64) into %fixed-stack.19, align 16)
;MIR64-DAG:      STFD killed $f25, 488, $x1 :: (store (s64) into %fixed-stack.18)
;MIR64-DAG:      STFD killed $f26, 496, $x1 :: (store (s64) into %fixed-stack.17, align 16)
;MIR64-DAG:      STFD killed $f27, 504, $x1 :: (store (s64) into %fixed-stack.16)
;MIR64-DAG:      STFD killed $f28, 512, $x1 :: (store (s64) into %fixed-stack.15, align 16)
;MIR64-DAG:      STFD killed $f29, 520, $x1 :: (store (s64) into %fixed-stack.14)
;MIR64-DAG:      STFD killed $f30, 528, $x1 :: (store (s64) into %fixed-stack.13, align 16)
;MIR64-DAG:      STFD killed $f31, 536, $x1 :: (store (s64) into %fixed-stack.12)
;MIR64-DAG:      STXVD2X killed $v20, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.11)
;MIR64-DAG:      STXVD2X killed $v21, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.10)
;MIR64-DAG:      STXVD2X killed $v22, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.9)
;MIR64-DAG:      STXVD2X killed $v23, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.8)
;MIR64-DAG:      STXVD2X killed $v24, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.7)
;MIR64-DAG:      STXVD2X killed $v25, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.6)
;MIR64-DAG:      STXVD2X killed $v26, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.5)
;MIR64-DAG:      STXVD2X killed $v27, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.4)
;MIR64-DAG:      STXVD2X killed $v28, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.3)
;MIR64-DAG:      STXVD2X killed $v29, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.2)
;MIR64-DAG:      STXVD2X killed $v30, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.1)
;MIR64-DAG:      STXVD2X killed $v31, $x1, killed $x{{[0-9]+}} :: (store (s128) into %fixed-stack.0)

; MIR64:         INLINEASM

; MIR64-DAG:     $v31 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.0)
; MIR64-DAG:     $v30 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.1)
; MIR64-DAG:     $v29 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.2)
; MIR64-DAG:     $v28 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.3)
; MIR64-DAG:     $v27 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.4)
; MIR64-DAG:     $v26 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.5)
; MIR64-DAG:     $v25 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.6)
; MIR64-DAG:     $v24 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.7)
; MIR64-DAG:     $v23 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.8)
; MIR64-DAG:     $v22 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.9)
; MIR64-DAG:     $v21 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.10)
; MIR64-DAG:     $v20 = LXVD2X $x1, killed $x{{[0-9]+}} :: (load (s128) from %fixed-stack.11)
; MIR64-DAG:     $f31 = LFD 536, $x1 :: (load (s64) from %fixed-stack.12)
; MIR64-DAG:     $f30 = LFD 528, $x1 :: (load (s64) from %fixed-stack.13, align 16)
; MIR64-DAG:     $f29 = LFD 520, $x1 :: (load (s64) from %fixed-stack.14)
; MIR64-DAG:     $f28 = LFD 512, $x1 :: (load (s64) from %fixed-stack.15, align 16)
; MIR64-DAG:     $f27 = LFD 504, $x1 :: (load (s64) from %fixed-stack.16)
; MIR64-DAG:     $f26 = LFD 496, $x1 :: (load (s64) from %fixed-stack.17, align 16)
; MIR64-DAG:     $f25 = LFD 488, $x1 :: (load (s64) from %fixed-stack.18)
; MIR64-DAG:     $f24 = LFD 480, $x1 :: (load (s64) from %fixed-stack.19, align 16)
; MIR64-DAG:     $f23 = LFD 472, $x1 :: (load (s64) from %fixed-stack.20)
; MIR64-DAG:     $f22 = LFD 464, $x1 :: (load (s64) from %fixed-stack.21, align 16)
; MIR64-DAG:     $f21 = LFD 456, $x1 :: (load (s64) from %fixed-stack.22)
; MIR64-DAG:     $f20 = LFD 448, $x1 :: (load (s64) from %fixed-stack.23, align 16)
; MIR64-DAG:     $f19 = LFD 440, $x1 :: (load (s64) from %fixed-stack.24)
; MIR64-DAG:     $f18 = LFD 432, $x1 :: (load (s64) from %fixed-stack.25, align 16)
; MIR64-DAG:     $f17 = LFD 424, $x1 :: (load (s64) from %fixed-stack.26)
; MIR64-DAG:     $f16 = LFD 416, $x1 :: (load (s64) from %fixed-stack.27, align 16)
; MIR64-DAG:     $f15 = LFD 408, $x1 :: (load (s64) from %fixed-stack.28)
; MIR64-DAG:     $f14 = LFD 400, $x1 :: (load (s64) from %fixed-stack.29, align 16)
; MIR64-DAG:     $x31 = LD 392, $x1 :: (load (s64) from %fixed-stack.30)
; MIR64-DAG:     $x30 = LD 384, $x1 :: (load (s64) from %fixed-stack.31, align 16)
; MIR64-DAG:     $x29 = LD 376, $x1 :: (load (s64) from %fixed-stack.32)
; MIR64-DAG:     $x28 = LD 368, $x1 :: (load (s64) from %fixed-stack.33, align 16)
; MIR64-DAG:     $x27 = LD 360, $x1 :: (load (s64) from %fixed-stack.34)
; MIR64-DAG:     $x26 = LD 352, $x1 :: (load (s64) from %fixed-stack.35, align 16)
; MIR64-DAG:     $x25 = LD 344, $x1 :: (load (s64) from %fixed-stack.36)
; MIR64-DAG:     $x24 = LD 336, $x1 :: (load (s64) from %fixed-stack.37, align 16)
; MIR64-DAG:     $x23 = LD 328, $x1 :: (load (s64) from %fixed-stack.38)
; MIR64-DAG:     $x22 = LD 320, $x1 :: (load (s64) from %fixed-stack.39, align 16)
; MIR64-DAG:     $x21 = LD 312, $x1 :: (load (s64) from %fixed-stack.40)
; MIR64-DAG:     $x20 = LD 304, $x1 :: (load (s64) from %fixed-stack.41, align 16)
; MIR64-DAG:     $x19 = LD 296, $x1 :: (load (s64) from %fixed-stack.42)
; MIR64-DAG:     $x18 = LD 288, $x1 :: (load (s64) from %fixed-stack.43, align 16)
; MIR64-DAG:     $x17 = LD 280, $x1 :: (load (s64) from %fixed-stack.44)
; MIR64-DAG:     $x16 = LD 272, $x1 :: (load (s64) from %fixed-stack.45, align 16)
; MIR64-DAG:     $x15 = LD 264, $x1 :: (load (s64) from %fixed-stack.46)
; MIR64-DAG:     $x14 = LD 256, $x1 :: (load (s64) from %fixed-stack.47, align 16)
; MIR64:         $x1 = ADDI8 $x1, 544
; MIR64-NEXT:    BLR8 implicit $lr8, implicit $rm

; ASM32-LABEL:  .fprs_gprs_vecregs:

; ASM32:          stwu 1, -448(1)
; ASM32-DAG:      li [[FIXEDSTACK11:[0-9]+]], 32
; ASM32-DAG:      stxvd2x 52, 1, [[FIXEDSTACK11]]                      # 16-byte Folded Spill
; ASM32-DAG:      li [[FIXEDSTACK10:[0-9]+]], 48
; ASM32-DAG:      stxvd2x 53, 1, [[FIXEDSTACK10]]                      # 16-byte Folded Spill
; ASM32-DAG:      li [[FIXEDSTACK9:[0-9]+]], 64
; ASM32-DAG:      stxvd2x 54, 1, [[FIXEDSTACK9]]                       # 16-byte Folded Spill
; ASM32-DAG:      li [[FIXEDSTACK8:[0-9]+]], 80
; ASM32-DAG:      stxvd2x 55, 1, [[FIXEDSTACK8]]                       # 16-byte Folded Spill
; ASM32-DAG:      li [[FIXEDSTACK7:[0-9]+]], 96
; ASM32-DAG:      stxvd2x 56, 1, [[FIXEDSTACK7]]                       # 16-byte Folded Spill
; ASM32-DAG:      li [[FIXEDSTACK6:[0-9]+]], 112
; ASM32-DAG:      stxvd2x 57, 1, [[FIXEDSTACK6]]                       # 16-byte Folded Spill
; ASM32-DAG:      li [[FIXEDSTACK5:[0-9]+]], 128
; ASM32-DAG:      stxvd2x 58, 1, [[FIXEDSTACK5]]                       # 16-byte Folded Spill
; ASM32-DAG:      li [[FIXEDSTACK4:[0-9]+]], 144
; ASM32-DAG:      stxvd2x 59, 1, [[FIXEDSTACK4]]                       # 16-byte Folded Spill
; ASM32-DAG:      li [[FIXEDSTACK3:[0-9]+]], 160
; ASM32-DAG:      stxvd2x 60, 1, [[FIXEDSTACK3]]                       # 16-byte Folded Spill
; ASM32-DAG:      li [[FIXEDSTACK2:[0-9]+]], 176
; ASM32-DAG:      stxvd2x 61, 1, [[FIXEDSTACK2]]                       # 16-byte Folded Spill
; ASM32-DAG:      li [[FIXEDSTACK1:[0-9]+]], 192
; ASM32-DAG:      stxvd2x 62, 1, [[FIXEDSTACK1]]                       # 16-byte Folded Spill
; ASM32-DAG:      li [[FIXEDSTACK0:[0-9]+]], 208
; ASM32-DAG:      stxvd2x 63, 1, [[FIXEDSTACK0]]                       # 16-byte Folded Spill
; ASM32-DAG:      stw 14, 232(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 15, 236(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 16, 240(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 17, 244(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 18, 248(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 19, 252(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 20, 256(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 21, 260(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 22, 264(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 23, 268(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 24, 272(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 25, 276(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 26, 280(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 27, 284(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 28, 288(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 29, 292(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 30, 296(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stw 31, 300(1)                          # 4-byte Folded Spill
; ASM32-DAG:      stfd 14, 304(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 15, 312(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 16, 320(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 17, 328(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 18, 336(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 19, 344(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 20, 352(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 21, 360(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 22, 368(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 23, 376(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 24, 384(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 25, 392(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 26, 400(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 27, 408(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 28, 416(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 29, 424(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 30, 432(1)                         # 8-byte Folded Spill
; ASM32-DAG:      stfd 31, 440(1)                         # 8-byte Folded Spill

; ASM32:          #APP
; ASM32-NEXT:     #NO_APP

; ASM32-DAG:      lxvd2x 63, 1, [[FIXEDSTACK0]]                        # 16-byte Folded Reload
; ASM32-DAG:      li [[FIXEDSTACK1:[0-9]+]], 192
; ASM32-DAG:      lxvd2x 62, 1, [[FIXEDSTACK1]]                        # 16-byte Folded Reload
; ASM32-DAG:      li [[FIXEDSTACK2:[0-9]+]], 176
; ASM32-DAG:      lxvd2x 61, 1, [[FIXEDSTACK2]]                        # 16-byte Folded Reload
; ASM32-DAG:      li [[FIXEDSTACK3:[0-9]+]], 160
; ASM32-DAG:      lxvd2x 60, 1, [[FIXEDSTACK3]]                        # 16-byte Folded Reload
; ASM32-DAG:      li [[FIXEDSTACK4:[0-9]+]], 144
; ASM32-DAG:      lxvd2x 59, 1, [[FIXEDSTACK4]]                        # 16-byte Folded Reload
; ASM32-DAG:      li [[FIXEDSTACK5:[0-9]+]], 128
; ASM32-DAG:      lxvd2x 58, 1, [[FIXEDSTACK5]]                        # 16-byte Folded Reload
; ASM32-DAG:      li [[FIXEDSTACK6:[0-9]+]], 112
; ASM32-DAG:      lxvd2x 57, 1, [[FIXEDSTACK6]]                        # 16-byte Folded Reload
; ASM32-DAG:      li [[FIXEDSTACK7:[0-9]+]], 96
; ASM32-DAG:      lxvd2x 56, 1, [[FIXEDSTACK7]]                        # 16-byte Folded Reload
; ASM32-DAG:      li [[FIXEDSTACK8:[0-9]+]], 80
; ASM32-DAG:      lxvd2x 55, 1, [[FIXEDSTACK8]]                        # 16-byte Folded Reload
; ASM32-DAG:      li [[FIXEDSTACK9:[0-9]+]], 64
; ASM32-DAG:      lxvd2x 54, 1, [[FIXEDSTACK9]]                        # 16-byte Folded Reload
; ASM32-DAG:      li [[FIXEDSTACK10:[0-9]+]], 48
; ASM32-DAG:      lxvd2x 53, 1, [[FIXEDSTACK10]]                        # 16-byte Folded Reload
; ASM32-DAG:      li [[FIXEDSTACK11:[0-9]+]], 32
; ASM32-DAG:      lxvd2x 52, 1, [[FIXEDSTACK11]]                        # 16-byte Folded Reload
; ASM32-DAG:      lfd 31, 440(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 30, 432(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 29, 424(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 28, 416(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 27, 408(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 26, 400(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 25, 392(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 24, 384(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 23, 376(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 22, 368(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 21, 360(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 20, 352(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 19, 344(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 18, 336(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 17, 328(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 16, 320(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 15, 312(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lfd 14, 304(1)                          # 8-byte Folded Reload
; ASM32-DAG:      lwz 31, 300(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 30, 296(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 29, 292(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 28, 288(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 27, 284(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 26, 280(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 25, 276(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 24, 272(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 23, 268(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 22, 264(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 21, 260(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 20, 256(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 19, 252(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 18, 248(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 17, 244(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 16, 240(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 15, 236(1)                          # 4-byte Folded Reload
; ASM32-DAG:      lwz 14, 232(1)                          # 4-byte Folded Reload

; ASM32:          addi 1, 1, 448
; ASM32-NEXT:     blr

; ASM64-LABEL:    .fprs_gprs_vecregs:

; ASM64:            stdu 1, -544(1)
; ASM64-DAG:        li [[FIXEDSTACK11:[0-9]+]], 64
; ASM64-DAG:        stxvd2x 52, 1, [[FIXEDSTACK11]]                       # 16-byte Folded Spill
; ASM64-DAG:        li [[FIXEDSTACK10:[0-9]+]], 80
; ASM64-DAG:        stxvd2x 53, 1, [[FIXEDSTACK10]]                       # 16-byte Folded Spill
; ASM64-DAG:        li [[FIXEDSTACK9:[0-9]+]], 96
; ASM64-DAG:        stxvd2x 54, 1, [[FIXEDSTACK9]]                        # 16-byte Folded Spill
; ASM64-DAG:        li [[FIXEDSTACK8:[0-9]+]], 112
; ASM64-DAG:        stxvd2x 55, 1, [[FIXEDSTACK8]]                        # 16-byte Folded Spill
; ASM64-DAG:        li [[FIXEDSTACK7:[0-9]+]], 128
; ASM64-DAG:        stxvd2x 56, 1, [[FIXEDSTACK7]]                        # 16-byte Folded Spill
; ASM64-DAG:        li [[FIXEDSTACK6:[0-9]+]], 144
; ASM64-DAG:        stxvd2x 57, 1, [[FIXEDSTACK6]]                        # 16-byte Folded Spill
; ASM64-DAG:        li [[FIXEDSTACK5:[0-9]+]], 160
; ASM64-DAG:        stxvd2x 58, 1, [[FIXEDSTACK5]]                        # 16-byte Folded Spill
; ASM64-DAG:        li [[FIXEDSTACK4:[0-9]+]], 176
; ASM64-DAG:        stxvd2x 59, 1, [[FIXEDSTACK4]]                        # 16-byte Folded Spill
; ASM64-DAG:        li [[FIXEDSTACK3:[0-9]+]], 192
; ASM64-DAG:        stxvd2x 60, 1, [[FIXEDSTACK3]]                        # 16-byte Folded Spill
; ASM64-DAG:        li [[FIXEDSTACK2:[0-9]+]], 208
; ASM64-DAG:        stxvd2x 61, 1, [[FIXEDSTACK2]]                        # 16-byte Folded Spill
; ASM64-DAG:        li [[FIXEDSTACK1:[0-9]+]], 224
; ASM64-DAG:        stxvd2x 62, 1, [[FIXEDSTACK1]]                        # 16-byte Folded Spill
; ASM64-DAG:        li [[FIXEDSTACK0:[0-9]+]], 240
; ASM64-DAG:        stxvd2x 63, 1, [[FIXEDSTACK0]]                        # 16-byte Folded Spill
; ASM64-DAG:        std 14, 256(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 15, 264(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 16, 272(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 17, 280(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 18, 288(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 19, 296(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 20, 304(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 21, 312(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 22, 320(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 23, 328(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 24, 336(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 25, 344(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 26, 352(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 27, 360(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 28, 368(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 29, 376(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 30, 384(1)                          # 8-byte Folded Spill
; ASM64-DAG:        std 31, 392(1)                          # 8-byte Folded Spill
; ASM64-DAG:        stfd 14, 400(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 15, 408(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 16, 416(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 17, 424(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 18, 432(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 19, 440(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 20, 448(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 21, 456(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 22, 464(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 23, 472(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 24, 480(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 25, 488(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 26, 496(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 27, 504(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 28, 512(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 29, 520(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 30, 528(1)                         # 8-byte Folded Spill
; ASM64-DAG:        stfd 31, 536(1)                         # 8-byte Folded Spill

; ASM64:            #APP
; ASM64-NEXT:       #NO_APP

; ASM64-DAG:        lxvd2x 63, 1, [[FIXEDSTACK0]]                         # 16-byte Folded Reload
; ASM64-DAG:        li [[FIXEDSTACK1:[0-9]+]], 224
; ASM64-DAG:        lxvd2x 62, 1, [[FIXEDSTACK1]]                         # 16-byte Folded Reload
; ASM64-DAG:        li [[FIXEDSTACK2:[0-9]+]], 208
; ASM64-DAG:        lxvd2x 61, 1, [[FIXEDSTACK2]]                         # 16-byte Folded Reload
; ASM64-DAG:        li [[FIXEDSTACK3:[0-9]+]], 192
; ASM64-DAG:        lxvd2x 60, 1, [[FIXEDSTACK3]]                         # 16-byte Folded Reload
; ASM64-DAG:        li [[FIXEDSTACK4:[0-9]+]], 176
; ASM64-DAG:        lxvd2x 59, 1, [[FIXEDSTACK4]]                         # 16-byte Folded Reload
; ASM64-DAG:        li [[FIXEDSTACK5:[0-9]+]], 160
; ASM64-DAG:        lxvd2x 58, 1, [[FIXEDSTACK5]]                         # 16-byte Folded Reload
; ASM64-DAG:        li [[FIXEDSTACK6:[0-9]+]], 144
; ASM64-DAG:        lxvd2x 57, 1, [[FIXEDSTACK6]]                         # 16-byte Folded Reload
; ASM64-DAG:        li [[FIXEDSTACK7:[0-9]+]], 128
; ASM64-DAG:        lxvd2x 56, 1, [[FIXEDSTACK7]]                         # 16-byte Folded Reload
; ASM64-DAG:        li [[FIXEDSTACK8:[0-9]+]], 112
; ASM64-DAG:        lxvd2x 55, 1, [[FIXEDSTACK8]]                         # 16-byte Folded Reload
; ASM64-DAG:        li [[FIXEDSTACK9:[0-9]+]], 96
; ASM64-DAG:        lxvd2x 54, 1, [[FIXEDSTACK9]]                         # 16-byte Folded Reload
; ASM64-DAG:        li [[FIXEDSTACK10:[0-9]+]], 80
; ASM64-DAG:        lxvd2x 53, 1, [[FIXEDSTACK10]]                        # 16-byte Folded Reload
; ASM64-DAG:        li [[FIXEDSTACK11:[0-9]+]], 64
; ASM64-DAG:        lxvd2x 52, 1, [[FIXEDSTACK11]]                        # 16-byte Folded Reload
; ASM64-DAG:        lfd 31, 536(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 30, 528(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 29, 520(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 28, 512(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 27, 504(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 26, 496(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 25, 488(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 24, 480(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 23, 472(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 22, 464(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 21, 456(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 20, 448(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 19, 440(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 18, 432(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 17, 424(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 16, 416(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 15, 408(1)                          # 8-byte Folded Reload
; ASM64-DAG:        lfd 14, 400(1)                          # 8-byte Folded Reload
; ASM64-DAG:        ld 31, 392(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 30, 384(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 29, 376(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 28, 368(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 27, 360(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 26, 352(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 25, 344(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 24, 336(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 23, 328(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 22, 320(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 21, 312(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 20, 304(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 19, 296(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 18, 288(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 17, 280(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 16, 272(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 15, 264(1)                           # 8-byte Folded Reload
; ASM64-DAG:        ld 14, 256(1)                           # 8-byte Folded Reload

; ASM64:            addi 1, 1, 544
; ASM64-NEXT:       blr

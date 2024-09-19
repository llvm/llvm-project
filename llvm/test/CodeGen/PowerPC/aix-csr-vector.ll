; RUN: llc -mtriple=powerpc-unknown-aix-xcoff -verify-machineinstrs -mcpu=pwr7 \
; RUN:     -mattr=+altivec -stop-after=prologepilog < %s | \
; RUN:   FileCheck --check-prefix=MIR32 %s

; RUN: llc -mtriple=powerpc-unknown-aix-xcoff -verify-machineinstrs \
; RUN:     -mcpu=pwr7 -mattr=+altivec < %s | \
; RUN:   FileCheck --check-prefix=ASM32 %s

; RUN: llc -mtriple=powerpc64-unknown-aix-xcoff -verify-machineinstrs \
; RUN:     -mcpu=pwr7 -mattr=+altivec -stop-after=prologepilog < %s | \
; RUN:   FileCheck --check-prefix=MIR64 %s

; RUN: llc -mtriple=powerpc64-unknown-aix-xcoff -verify-machineinstrs \
; RUN:     -mcpu=pwr7 -mattr=+altivec < %s | \
; RUN:   FileCheck --check-prefix=ASM64 %s

define dso_local void @vec_regs() {
  entry:
    call void asm sideeffect "", "~{v13},~{v20},~{v26},~{v31}"()
      ret void
}

; MIR32-LABEL:   name:            vec_regs

; MIR32:         fixedStack:      []
; MIR32-NOT:     STXVD2X killed $v20
; MIR32-NOT:     STXVD2X killed $v26
; MIR32-NOT:     STXVD2X killed $v31
; MIR32-LABEL:   INLINEASM
; MIR32-NOT:     $v20 = LXVD2X
; MIR32-NOT:     $v26 = LXVD2X
; MIR32-NOT:     $v31 = LXVD2X
; MIR32:         BLR implicit $lr, implicit $rm

; MIR64-LABEL:   name:            vec_regs

; MIR64:         fixedStack:      []
; MIR64-NOT:     STXVD2X killed $v20
; MIR64-NOT:     STXVD2X killed $v26
; MIR64-NOT:     STXVD2X killed $v31
; MIR64-LABEL:   INLINEASM
; MIR64-NOT:     $v20 = LXVD2X
; MIR64-NOT:     $v26 = LXVD2X
; MIR64-NOT:     $v31 = LXVD2X
; MIR64:         BLR8 implicit $lr8, implicit $rm

; ASM32-LABEL:   .vec_regs:

; ASM32-NOT:     20
; ASM32-NOT:     26
; ASM32-NOT:     31
; ASM32-DAG:     #APP
; ASM32-DAG:     #NO_APP
; ASM32:         blr

; ASM64-LABEL:   .vec_regs:

; ASM64-NOT:     20
; ASM64-NOT:     26
; ASM64-NOT:     31
; ASM64-DAG:     #APP
; ASM64-DAG:     #NO_APP
; ASM64:         blr

define dso_local void @fprs_gprs_vecregs() {
    call void asm sideeffect "", "~{r25},~{r28},~{r31},~{f21},~{f25},~{f31},~{v20},~{v26},~{v31}"()
      ret void
}

; MIR32-LABEL:   name:            fprs_gprs_vecregs

; MIR32: liveins: $r25, $r26, $r27, $r28, $r29, $r30, $r31, $f21, $f22, $f23, $f24, $f25, $f26, $f27, $f28, $f29, $f30, $f31

; MIR32-NOT:     STXVD2X killed $v20
; MIR32-NOT:     STXVD2X killed $v26
; MIR32-NOT:     STXVD2X killed $v31
; MIR32-DAG:     STW killed $r25, -116, $r1 :: (store (s32) into %fixed-stack.17)
; MIR32-DAG:     STW killed $r26, -112, $r1 :: (store (s32) into %fixed-stack.16, align 8)
; MIR32-DAG:     STW killed $r27, -108, $r1 :: (store (s32) into %fixed-stack.15)
; MIR32-DAG:     STW killed $r28, -104, $r1 :: (store (s32) into %fixed-stack.14, align 16)
; MIR32-DAG:     STW killed $r29, -100, $r1 :: (store (s32) into %fixed-stack.13)
; MIR32-DAG:     STW killed $r30, -96, $r1 :: (store (s32) into %fixed-stack.12, align 8)
; MIR32-DAG:     STW killed $r31, -92, $r1 :: (store (s32) into %fixed-stack.11)
; MIR32-DAG:     STFD killed $f21, -88, $r1 :: (store (s64) into %fixed-stack.10)
; MIR32-DAG:     STFD killed $f22, -80, $r1 :: (store (s64) into %fixed-stack.9, align 16)
; MIR32-DAG:     STFD killed $f23, -72, $r1 :: (store (s64) into %fixed-stack.8)
; MIR32-DAG:     STFD killed $f24, -64, $r1 :: (store (s64) into %fixed-stack.7, align 16)
; MIR32-DAG:     STFD killed $f25, -56, $r1 :: (store (s64) into %fixed-stack.6)
; MIR32-DAG:     STFD killed $f26, -48, $r1 :: (store (s64) into %fixed-stack.5, align 16)
; MIR32-DAG:     STFD killed $f27, -40, $r1 :: (store (s64) into %fixed-stack.4)
; MIR32-DAG:     STFD killed $f28, -32, $r1 :: (store (s64) into %fixed-stack.3, align 16)
; MIR32-DAG:     STFD killed $f29, -24, $r1 :: (store (s64) into %fixed-stack.2)
; MIR32-DAG:     STFD killed $f30, -16, $r1 :: (store (s64) into %fixed-stack.1, align 16)
; MIR32-DAG:     STFD killed $f31, -8, $r1 :: (store (s64) into %fixed-stack.0)

; MIR32-LABEL:   INLINEASM

; MIR32-NOT:     $v20 = LXVD2X
; MIR32-NOT:     $v26 = LXVD2X
; MIR32-NOT:     $v31 = LXVD2X
; MIR32-DAG:     $f31 = LFD -8, $r1 :: (load (s64) from %fixed-stack.0)
; MIR32-DAG:     $f30 = LFD -16, $r1 :: (load (s64) from %fixed-stack.1, align 16)
; MIR32-DAG:     $f29 = LFD -24, $r1 :: (load (s64) from %fixed-stack.2)
; MIR32-DAG:     $f28 = LFD -32, $r1 :: (load (s64) from %fixed-stack.3, align 16)
; MIR32-DAG:     $f27 = LFD -40, $r1 :: (load (s64) from %fixed-stack.4)
; MIR32-DAG:     $f26 = LFD -48, $r1 :: (load (s64) from %fixed-stack.5, align 16)
; MIR32-DAG:     $f25 = LFD -56, $r1 :: (load (s64) from %fixed-stack.6)
; MIR32-DAG:     $f24 = LFD -64, $r1 :: (load (s64) from %fixed-stack.7, align 16)
; MIR32-DAG:     $f23 = LFD -72, $r1 :: (load (s64) from %fixed-stack.8)
; MIR32-DAG:     $f22 = LFD -80, $r1 :: (load (s64) from %fixed-stack.9, align 16)
; MIR32-DAG:     $f21 = LFD -88, $r1 :: (load (s64) from %fixed-stack.10)
; MIR32-DAG:     $r31 = LWZ -92, $r1 :: (load (s32) from %fixed-stack.11)
; MIR32-DAG:     $r30 = LWZ -96, $r1 :: (load (s32) from %fixed-stack.12, align 8)
; MIR32-DAG:     $r29 = LWZ -100, $r1 :: (load (s32) from %fixed-stack.13)
; MIR32-DAG:     $r28 = LWZ -104, $r1 :: (load (s32) from %fixed-stack.14, align 16)
; MIR32-DAG:     $r27 = LWZ -108, $r1 :: (load (s32) from %fixed-stack.15)
; MIR32-DAG:     $r26 = LWZ -112, $r1 :: (load (s32) from %fixed-stack.16, align 8)
; MIR32-DAG:     $r25 = LWZ -116, $r1 :: (load (s32) from %fixed-stack.17)
; MIR32:         BLR implicit $lr, implicit $rm

; MIR64-LABEL:   name:            fprs_gprs_vecregs

; MIR64: liveins: $x25, $x26, $x27, $x28, $x29, $x30, $x31, $f21, $f22, $f23, $f24, $f25, $f26, $f27, $f28, $f29, $f30, $f31

; MIR64-NOT:     STXVD2X killed $v20
; MIR64-NOT:     STXVD2X killed $v26
; MIR64-NOT:     STXVD2X killed $v31
; MIR64-DAG:     STD killed $x25, -144, $x1 :: (store (s64) into %fixed-stack.17)
; MIR64-DAG:     STD killed $x26, -136, $x1 :: (store (s64) into %fixed-stack.16, align 16)
; MIR64-DAG:     STD killed $x27, -128, $x1 :: (store (s64) into %fixed-stack.15)
; MIR64-DAG:     STD killed $x28, -120, $x1 :: (store (s64) into %fixed-stack.14, align 16)
; MIR64-DAG:     STD killed $x29, -112, $x1 :: (store (s64) into %fixed-stack.13)
; MIR64-DAG:     STD killed $x30, -104, $x1 :: (store (s64) into %fixed-stack.12, align 16)
; MIR64-DAG:     STD killed $x31, -96, $x1 :: (store (s64) into %fixed-stack.11)
; MIR64-DAG:     STFD killed $f21, -88, $x1 :: (store (s64) into %fixed-stack.10)
; MIR64-DAG:     STFD killed $f22, -80, $x1 :: (store (s64) into %fixed-stack.9, align 16)
; MIR64-DAG:     STFD killed $f23, -72, $x1 :: (store (s64) into %fixed-stack.8)
; MIR64-DAG:     STFD killed $f24, -64, $x1 :: (store (s64) into %fixed-stack.7, align 16)
; MIR64-DAG:     STFD killed $f25, -56, $x1 :: (store (s64) into %fixed-stack.6)
; MIR64-DAG:     STFD killed $f26, -48, $x1 :: (store (s64) into %fixed-stack.5, align 16)
; MIR64-DAG:     STFD killed $f27, -40, $x1 :: (store (s64) into %fixed-stack.4)
; MIR64-DAG:     STFD killed $f28, -32, $x1 :: (store (s64) into %fixed-stack.3, align 16)
; MIR64-DAG:     STFD killed $f29, -24, $x1 :: (store (s64) into %fixed-stack.2)
; MIR64-DAG:     STFD killed $f30, -16, $x1 :: (store (s64) into %fixed-stack.1, align 16)
; MIR64-DAG:     STFD killed $f31, -8, $x1 :: (store (s64) into %fixed-stack.0)

; MIR64-LABEL:   INLINEASM

; MIR64-NOT:     $v20 = LXVD2X
; MIR64-NOT:     $v26 = LXVD2X
; MIR64-NOT:     $v31 = LXVD2X
; MIR64-DAG:     $f31 = LFD -8, $x1 :: (load (s64) from %fixed-stack.0)
; MIR64-DAG:     $f30 = LFD -16, $x1 :: (load (s64) from %fixed-stack.1, align 16)
; MIR64-DAG:     $f29 = LFD -24, $x1 :: (load (s64) from %fixed-stack.2)
; MIR64-DAG:     $f28 = LFD -32, $x1 :: (load (s64) from %fixed-stack.3, align 16)
; MIR64-DAG:     $f27 = LFD -40, $x1 :: (load (s64) from %fixed-stack.4)
; MIR64-DAG:     $f26 = LFD -48, $x1 :: (load (s64) from %fixed-stack.5, align 16)
; MIR64-DAG:     $f25 = LFD -56, $x1 :: (load (s64) from %fixed-stack.6)
; MIR64-DAG:     $f24 = LFD -64, $x1 :: (load (s64) from %fixed-stack.7, align 16)
; MIR64-DAG:     $f23 = LFD -72, $x1 :: (load (s64) from %fixed-stack.8)
; MIR64-DAG:     $f22 = LFD -80, $x1 :: (load (s64) from %fixed-stack.9, align 16)
; MIR64-DAG:     $f21 = LFD -88, $x1 :: (load (s64) from %fixed-stack.10)
; MIR64-DAG:     $x31 = LD -96, $x1 :: (load (s64) from %fixed-stack.11)
; MIR64-DAG:     $x30 = LD -104, $x1 :: (load (s64) from %fixed-stack.12, align 16)
; MIR64-DAG:     $x29 = LD -112, $x1 :: (load (s64) from %fixed-stack.13)
; MIR64-DAG:     $x28 = LD -120, $x1 :: (load (s64) from %fixed-stack.14, align 16)
; MIR64-DAG:     $x27 = LD -128, $x1 :: (load (s64) from %fixed-stack.15)
; MIR64-DAG:     $x26 = LD -136, $x1 :: (load (s64) from %fixed-stack.16, align 16)
; MIR64-DAG:     $x25 = LD -144, $x1 :: (load (s64) from %fixed-stack.17)

; MIR64:         BLR8 implicit $lr8, implicit $rm

;; We don't have -ppc-full-reg-names on AIX so can't reliably check-not for
;; only vector registers numbers in this case.

; ASM32-LABEL:   .fprs_gprs_vecregs:

; ASM32-DAG:   stw 25, -116(1)                         # 4-byte Folded Spill
; ASM32-DAG:   stw 26, -112(1)                         # 4-byte Folded Spill
; ASM32-DAG:   stw 27, -108(1)                         # 4-byte Folded Spill
; ASM32-DAG:   stw 28, -104(1)                         # 4-byte Folded Spill
; ASM32-DAG:   stw 29, -100(1)                         # 4-byte Folded Spill
; ASM32-DAG:   stw 30, -96(1)                          # 4-byte Folded Spill
; ASM32-DAG:   stw 31, -92(1)                          # 4-byte Folded Spill
; ASM32-DAG:   stfd 21, -88(1)                         # 8-byte Folded Spill
; ASM32-DAG:   stfd 22, -80(1)                         # 8-byte Folded Spill
; ASM32-DAG:   stfd 23, -72(1)                         # 8-byte Folded Spill
; ASM32-DAG:   stfd 24, -64(1)                         # 8-byte Folded Spill
; ASM32-DAG:   stfd 25, -56(1)                         # 8-byte Folded Spill
; ASM32-DAG:   stfd 26, -48(1)                         # 8-byte Folded Spill
; ASM32-DAG:   stfd 27, -40(1)                         # 8-byte Folded Spill
; ASM32-DAG:   stfd 28, -32(1)                         # 8-byte Folded Spill
; ASM32-DAG:   stfd 29, -24(1)                         # 8-byte Folded Spill
; ASM32-DAG:   stfd 30, -16(1)                         # 8-byte Folded Spill
; ASM32-DAG:   stfd 31, -8(1)                          # 8-byte Folded Spill
; ASM32:       #APP
; ASM32-NEXT:  #NO_APP
; ASM32-DAG:   lfd 31, -8(1)                           # 8-byte Folded Reload
; ASM32-DAG:   lfd 30, -16(1)                          # 8-byte Folded Reload
; ASM32-DAG:   lfd 29, -24(1)                          # 8-byte Folded Reload
; ASM32-DAG:   lfd 28, -32(1)                          # 8-byte Folded Reload
; ASM32-DAG:   lfd 27, -40(1)                          # 8-byte Folded Reload
; ASM32-DAG:   lfd 26, -48(1)                          # 8-byte Folded Reload
; ASM32-DAG:   lfd 25, -56(1)                          # 8-byte Folded Reload
; ASM32-DAG:   lfd 24, -64(1)                          # 8-byte Folded Reload
; ASM32-DAG:   lfd 23, -72(1)                          # 8-byte Folded Reload
; ASM32-DAG:   lfd 22, -80(1)                          # 8-byte Folded Reload
; ASM32-DAG:   lfd 21, -88(1)                          # 8-byte Folded Reload
; ASM32-DAG:   lwz 31, -92(1)                          # 4-byte Folded Reload
; ASM32-DAG:   lwz 30, -96(1)                          # 4-byte Folded Reload
; ASM32-DAG:   lwz 29, -100(1)                         # 4-byte Folded Reload
; ASM32-DAG:   lwz 28, -104(1)                         # 4-byte Folded Reload
; ASM32-DAG:   lwz 27, -108(1)                         # 4-byte Folded Reload
; ASM32-DAG:   lwz 26, -112(1)                         # 4-byte Folded Reload
; ASM32-DAG:   lwz 25, -116(1)                         # 4-byte Folded Reload
; ASM32:         blr

; ASM64-LABEL:    .fprs_gprs_vecregs:

; ASM64-DAG:     std 25, -144(1)                         # 8-byte Folded Spill
; ASM64-DAG:     std 26, -136(1)                         # 8-byte Folded Spill
; ASM64-DAG:     std 27, -128(1)                         # 8-byte Folded Spill
; ASM64-DAG:     std 28, -120(1)                         # 8-byte Folded Spill
; ASM64-DAG:     std 29, -112(1)                         # 8-byte Folded Spill
; ASM64-DAG:     std 30, -104(1)                         # 8-byte Folded Spill
; ASM64-DAG:     std 31, -96(1)                          # 8-byte Folded Spill
; ASM64-DAG:     stfd 21, -88(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 22, -80(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 23, -72(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 24, -64(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 25, -56(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 26, -48(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 27, -40(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 28, -32(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 29, -24(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 30, -16(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 31, -8(1)                          # 8-byte Folded Spill
; ASM64:         #APP
; ASM64-NEXT:    #NO_APP
; ASM64-DAG:     lfd 31, -8(1)                           # 8-byte Folded Reload
; ASM64-DAG:     lfd 30, -16(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 29, -24(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 28, -32(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 27, -40(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 26, -48(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 25, -56(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 24, -64(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 23, -72(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 22, -80(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 21, -88(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 31, -96(1)                           # 8-byte Folded Reload
; ASM64-DAG:     ld 30, -104(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 29, -112(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 28, -120(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 27, -128(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 26, -136(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 25, -144(1)                          # 8-byte Folded Reload

; ASM64:         blr

define dso_local void @all_fprs_and_vecregs() {
    call void asm sideeffect "", "~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6}~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}"()
      ret void
}

;; Check that reserved vectors are not used.
; MIR32-LABEL:   all_fprs_and_vecregs

; MIR32-NOT:     $v20
; MIR32-NOT:     $v21
; MIR32-NOT:     $v22
; MIR32-NOT:     $v23
; MIR32-NOT:     $v24
; MIR32-NOT:     $v25
; MIR32-NOT:     $v26
; MIR32-NOT:     $v27
; MIR32-NOT:     $v28
; MIR32-NOT:     $v29
; MIR32-NOT:     $v30
; MIR32-NOT:     $v31

; MIR64-LABEL:   all_fprs_and_vecregs

; MIR64-NOT:     $v20
; MIR64-NOT:     $v21
; MIR64-NOT:     $v22
; MIR64-NOT:     $v23
; MIR64-NOT:     $v24
; MIR64-NOT:     $v25
; MIR64-NOT:     $v26
; MIR64-NOT:     $v27
; MIR64-NOT:     $v28
; MIR64-NOT:     $v29
; MIR64-NOT:     $v30
; MIR64-NOT:     $v31

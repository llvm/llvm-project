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
    call void asm sideeffect "", "~{r14},~{r25},~{r31},~{f14},~{f21},~{f31},~{v20},~{v26},~{v31}"()
      ret void
}

; MIR32-LABEL:   name:            fprs_gprs_vecregs

; MIR32:         fixedStack:

; MIR32:         liveins: $r14, $r25, $r31, $f14, $f21, $f31

; MIR32-NOT:     STXVD2X killed $v20
; MIR32-NOT:     STXVD2X killed $v26
; MIR32-NOT:     STXVD2X killed $v31
; MIR32-DAG:     STW killed $r14, -216, $r1 :: (store (s32) into %fixed-stack.5, align 8)
; MIR32-DAG:     STW killed $r25, -172, $r1 :: (store (s32) into %fixed-stack.4)
; MIR32-DAG:     STW killed $r31, -148, $r1 :: (store (s32) into %fixed-stack.3)
; MIR32-DAG:     STFD killed $f14, -144, $r1 :: (store (s64) into %fixed-stack.2, align 16)
; MIR32-DAG:     STFD killed $f21, -88, $r1 :: (store (s64) into %fixed-stack.1)
; MIR32-DAG:     STFD killed $f31, -8, $r1 :: (store (s64) into %fixed-stack.0)

; MIR32-LABEL:   INLINEASM

; MIR32-NOT:     $v20 = LXVD2X
; MIR32-NOT:     $v26 = LXVD2X
; MIR32-NOT:     $v31 = LXVD2X
; MIR32-DAG:     $r14 = LWZ -216, $r1 :: (load (s32) from %fixed-stack.5, align 8)
; MIR32-DAG:     $r25 = LWZ -172, $r1 :: (load (s32) from %fixed-stack.4)
; MIR32-DAG:     $r31 = LWZ -148, $r1 :: (load (s32) from %fixed-stack.3)
; MIR32-DAG:     $f14 = LFD -144, $r1 :: (load (s64) from %fixed-stack.2, align 16)
; MIR32-DAG:     $f21 = LFD -88, $r1 :: (load (s64) from %fixed-stack.1)
; MIR32-DAG:     $f31 = LFD -8, $r1 :: (load (s64) from %fixed-stack.0)
; MIR32-DAG:     BLR implicit $lr, implicit $rm

; MIR64-LABEL:   name:            fprs_gprs_vecregs

; MIR64:         fixedStack:

; MIR64:         liveins: $x14, $x25, $x31, $f14, $f21, $f31

; MIR64-NOT:     STXVD2X killed $v20
; MIR64-NOT:     STXVD2X killed $v26
; MIR64-NOT:     STXVD2X killed $v31
; MIR64-DAG:     STD killed $x14, -288, $x1 :: (store (s64) into %fixed-stack.5, align 16)
; MIR64-DAG:     STD killed $x25, -200, $x1 :: (store (s64) into %fixed-stack.4)
; MIR64-DAG:     STD killed $x31, -152, $x1 :: (store (s64) into %fixed-stack.3)
; MIR64-DAG:     STFD killed $f14, -144, $x1 :: (store (s64) into %fixed-stack.2, align 16)
; MIR64-DAG:     STFD killed $f21, -88, $x1 :: (store (s64) into %fixed-stack.1)
; MIR64-DAG:     STFD killed $f31, -8, $x1 :: (store (s64) into %fixed-stack.0)

; MIR64-LABEL:   INLINEASM

; MIR64-NOT:     $v20 = LXVD2X
; MIR64-NOT:     $v26 = LXVD2X
; MIR64-NOT:     $v31 = LXVD2X
; MIR64-DAG:     $x14 = LD -288, $x1 :: (load (s64) from %fixed-stack.5, align 16)
; MIR64-DAG:     $x25 = LD -200, $x1 :: (load (s64) from %fixed-stack.4)
; MIR64-DAG:     $x31 = LD -152, $x1 :: (load (s64) from %fixed-stack.3)
; MIR64-DAG:     $f14 = LFD -144, $x1 :: (load (s64) from %fixed-stack.2, align 16)
; MIR64-DAG:     $f21 = LFD -88, $x1 :: (load (s64) from %fixed-stack.1)
; MIR64-DAG:     $f31 = LFD -8, $x1 :: (load (s64) from %fixed-stack.0)
; MIR64:         BLR8 implicit $lr8, implicit $rm

;; We don't have -ppc-full-reg-names on AIX so can't reliably check-not for
;; only vector registers numbers in this case.

; ASM32-LABEL:   .fprs_gprs_vecregs:

; ASM32-DAG:     stw 14, -216(1)                         # 4-byte Folded Spill
; ASM32-DAG:     stw 25, -172(1)                         # 4-byte Folded Spill
; ASM32-DAG:     stw 31, -148(1)                         # 4-byte Folded Spill
; ASM32-DAG:     stfd 14, -144(1)                        # 8-byte Folded Spill
; ASM32-DAG:     stfd 21, -88(1)                         # 8-byte Folded Spill
; ASM32-DAG:     stfd 31, -8(1)                          # 8-byte Folded Spill
; ASM32-DAG:     #APP
; ASM32-DAG:     #NO_APP
; ASM32-DAG:     lfd 31, -8(1)                           # 8-byte Folded Reload
; ASM32-DAG:     lfd 21, -88(1)                          # 8-byte Folded Reload
; ASM32-DAG:     lfd 14, -144(1)                         # 8-byte Folded Reload
; ASM32-DAG:     lwz 31, -148(1)                         # 4-byte Folded Reload
; ASM32-DAG:     lwz 25, -172(1)                         # 4-byte Folded Reload
; ASM32-DAG:     lwz 14, -216(1)                         # 4-byte Folded Reload
; ASM32:         blr

; ASM64-LABEL:    .fprs_gprs_vecregs:

; ASM64-DAG:     std 14, -288(1)                         # 8-byte Folded Spill
; ASM64-DAG:     std 25, -200(1)                         # 8-byte Folded Spill
; ASM64-DAG:     std 31, -152(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 14, -144(1)                        # 8-byte Folded Spill
; ASM64-DAG:     stfd 21, -88(1)                         # 8-byte Folded Spill
; ASM64-DAG:     stfd 31, -8(1)                          # 8-byte Folded Spill
; ASM64-DAG:     #APP
; ASM64-DAG:     #NO_APP
; ASM64-DAG:     lfd 31, -8(1)                           # 8-byte Folded Reload
; ASM64-DAG:     lfd 21, -88(1)                          # 8-byte Folded Reload
; ASM64-DAG:     lfd 14, -144(1)                         # 8-byte Folded Reload
; ASM64-DAG:     ld 31, -152(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 25, -200(1)                          # 8-byte Folded Reload
; ASM64-DAG:     ld 14, -288(1)                          # 8-byte Folded Reload
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

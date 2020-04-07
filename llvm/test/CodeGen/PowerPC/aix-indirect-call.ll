; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc-ibm-aix-xcoff -stop-after=machine-cp < %s | \
; RUN: FileCheck --check-prefixes=CHECKMIR,MIR32 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=CHECKASM,ASMOBJ32,ASM32 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp < %s | \
; RUN: FileCheck --check-prefixes=CHECKMIR,MIR64 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=CHECKASM,ASM64 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple powerpc-ibm-aix-xcoff  -filetype=obj < %s -o %t
; RUN: llvm-objdump -d %t | FileCheck \
; RUN: --check-prefixes=CHECKOBJ,ASMOBJ32,OBJ32 %s

define signext i32 @callThroughPtr(i32 ()* nocapture) {
  %2 = tail call signext i32 %0()
  ret i32 %2
}

; CHECKMIR:   name:            callThroughPtr

; MIR32:      liveins: $r3
; MIR32:      ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; MIR32-DAG:  STW $r2, 20, $r1
; MIR32-DAG:  renamable $r11 = LWZ 8, renamable $r3 :: (dereferenceable invariant load 4 from %ir.0 + 8)
; MIR32-DAG:  renamable $[[REG:r[0-9]+]] = LWZ 0, renamable $r3 :: (dereferenceable invariant load 4 from %ir.0)
; MIR32-DAG:  $r2 = LWZ 4, killed renamable $r3 :: (dereferenceable invariant load 4 from %ir.0 + 4)
; MIR32-DAG:  MTCTR killed renamable $[[REG]], implicit-def $ctr
; MIR32-NEXT: BCTRL_LWZinto_toc 20, $r1, csr_aix32, implicit-def dead $lr, implicit-def dead $r2, implicit $ctr, implicit $rm, implicit $r11, implicit $r2, implicit-def $r1, implicit-def $r3
; MIR32-NEXT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; MIR64:      liveins: $x3
; MIR64:      ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; MIR64-DAG:  STD $x2, 40, $x1 :: (store 8 into stack + 40)
; MIR64-DAG:  renamable $x11 = LD 16, renamable $x3 :: (dereferenceable invariant load 8 from %ir.0 + 16)
; MIR64-DAG:  renamable $[[REG:x[0-9]+]] = LD 0, renamable $x3 :: (dereferenceable invariant load 8 from %ir.0)
; MIR64-DAG:  $x2 = LD 8, killed renamable $x3 :: (dereferenceable invariant load 8 from %ir.0 + 8)
; MIR64-DAG:  MTCTR8 killed renamable $[[REG]], implicit-def $ctr8
; MIR64-NEXT: BCTRL8_LDinto_toc 40, $x1, csr_ppc64, implicit-def dead $lr8, implicit-def dead $x2, implicit $ctr8, implicit $rm, implicit $x11, implicit $x2, implicit-def $r1, implicit-def $x3
; MIR64-NEXT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .callThroughPtr:

; ASM32:         stwu 1, -64(1)
; ASM32-DAG:     lwz [[REG:[0-9]+]], 0(3)
; ASM32-DAG:     stw 2, 20(1)
; ASM32-DAG:     mtctr [[REG]]
; ASM32-DAG:     lwz 11, 8(3)
; ASM32-DAG:     lwz 2, 4(3)
; ASM32-NEXT:    bctrl
; ASM32-NEXT:    lwz 2, 20(1)
; ASM32-NEXT:    addi 1, 1, 64

; ASM64:            stdu 1, -112(1)
; ASM64-DAG:        ld [[REG:[0-9]+]], 0(3)
; ASM64-DAG:        std 2, 40(1)
; ASM64-DAG:        mtctr [[REG]]
; ASM64-DAG:        ld 11, 16(3)
; ASM64-DAG:        ld 2, 8(3)
; ASM64-NEXT:       bctrl
; ASM64-NEXT:       ld 2, 40(1)
; ASM64-NEXT:       addi 1, 1, 112

; OBJ32-LABEL: .text:
; OBJ32:                      stwu 1, -64(1)
; OBJ32-DAG:                  lwz [[REG:[0-9]+]], 0(3)
; OBJ32-DAG:                  stw 2, 20(1)
; OBJ32-DAG:                  mtctr [[REG]]
; OBJ32-DAG:                  lwz 11, 8(3)
; OBJ32-DAG:                  lwz 2, 4(3)
; OBJ32-NEXT:    4e 80 04 21  bctrl
; OBJ32-NEXT:    80 41 00 14  lwz 2, 20(1)
; OBJ32-NEXT:                 addi 1, 1, 64

define void @callThroughPtrWithArgs(void (i32, i16, i64)* nocapture) {
  tail call void %0(i32 signext 1, i16 zeroext 2, i64 3)
  ret void
}

; CHECKMIR:   name:            callThroughPtrWithArgs

; MIR32:      liveins: $r3
; MIR32:      ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; MIR32-DAG:  renamable $[[REG:r[0-9]+]] = LWZ 0, renamable $r3 :: (dereferenceable invariant load 4 from %ir.0)
; MIR32-DAG:  MTCTR killed renamable $[[REG]], implicit-def $ctr
; MIR32-DAG:  STW $r2, 20, $r1 :: (store 4 into stack + 20)
; MIR32-DAG:  renamable $r11 = LWZ 8, renamable $r3 :: (dereferenceable invariant load 4 from %ir.0 + 8)
; MIR32-DAG:  $r2 = LWZ 4, killed renamable $r3 :: (dereferenceable invariant load 4 from %ir.0 + 4)
; MIR32-DAG:  $r3 = LI 1
; MIR32-DAG:  $r4 = LI 2
; MIR32-DAG:  $r5 = LI 0
; MIR32-DAG:  $r6 = LI 3
; MIR32-NEXT: BCTRL_LWZinto_toc 20, $r1, csr_aix32, implicit-def dead $lr, implicit-def dead $r2, implicit $ctr, implicit $rm, implicit $r11, implicit $r3, implicit $r4, implicit $r5, implicit $r6, implicit $r2, implicit-def $r1
; MIR32-NEXT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; MIR64:      liveins: $x3
; MIR64:      ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; MIR64-DAG:  renamable $[[REG:x[0-9]+]] = LD 0, renamable $x3 :: (dereferenceable invariant load 8 from %ir.0)
; MIR64-DAG:  MTCTR8 killed renamable $[[REG]], implicit-def $ctr8
; MIR64-DAG:  STD $x2, 40, $x1 :: (store 8 into stack + 40)
; MIR64-DAG:  renamable $x11 = LD 16, renamable $x3 :: (dereferenceable invariant load 8 from %ir.0 + 16)
; MIR64-DAG:  $x2 = LD 8, killed renamable $x3 :: (dereferenceable invariant load 8 from %ir.0 + 8)
; MIR64-DAG:  $x3 = LI8 1
; MIR64-DAG:  $x4 = LI8 2
; MIR64-DAG:  $x5 = LI8 3
; MIR64-NEXT: BCTRL8_LDinto_toc 40, $x1, csr_ppc64, implicit-def dead $lr8, implicit-def dead $x2, implicit $ctr8, implicit $rm, implicit $x11, implicit $x3, implicit $x4, implicit $x5, implicit $x2, implicit-def $r1
; MIR64-NEXT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .callThroughPtrWithArgs:
; CHECKOBJ-LABEL: <.callThroughPtrWithArgs>:

; ASMOBJ32:      stwu 1, -64(1)
; ASMOBJ32-DAG:  lwz [[REG:[0-9]+]], 0(3)
; ASMOBJ32-DAG:  li 5, 0
; ASMOBJ32-DAG:  li 6, 3
; ASMOBJ32-DAG:  stw 2, 20(1)
; ASMOBJ32-DAG:  mtctr [[REG]]
; ASMOBJ32-DAG:  li 4, 2
; ASMOBJ32-DAG:  lwz 11, 8(3)
; ASMOBJ32-DAG:  lwz 2, 4(3)
; ASMOBJ32-DAG:  li 3, 1
; ASMOBJ32-NEXT: bctrl
; ASMOBJ32-NEXT: lwz 2, 20(1)
; ASMOBJ32-NEXT: addi 1, 1, 64

; ASM64:            stdu 1, -112(1)
; ASM64-DAG:        ld [[REG:[0-9]+]], 0(3)
; ASM64-DAG:        li 5, 3
; ASM64-DAG:        std 2, 40(1)
; ASM64-DAG:        mtctr [[REG]]
; ASM64-DAG:        li 4, 2
; ASM64-DAG:        ld 11, 16(3)
; ASM64-DAG:        ld 2, 8(3)
; ASM64-DAG:        li 3, 1
; ASM64-NEXT:       bctrl
; ASM64-NEXT:       ld 2, 40(1)
; ASM64-NEXT:       addi 1, 1, 112

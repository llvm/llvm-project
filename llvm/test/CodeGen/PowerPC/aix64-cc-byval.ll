; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp -verify-machineinstrs < %s | \
; RUN: FileCheck --check-prefixes=CHECK,64BIT %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=CHECKASM,ASM64PWR4 %s

%struct.S5 = type { [5 x i8] }

@gS5 = external global %struct.S5, align 1

define void @call_test_byval_5Byte() {
entry:
  call void @test_byval_5Byte(%struct.S5* byval(%struct.S5) align 1 @gS5)
  ret void
}

declare void @test_byval_5Byte(%struct.S5* byval(%struct.S5) align 1)

; CHECK-LABEL: name: call_test_byval_5Byte{{.*}}

; CHECKASM-LABEL: .call_test_byval_5Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS5, $x2 :: (load 8 from got)
; 64BIT-DAG:   renamable $x[[REG1:[0-9]+]] = LWZ8 0, killed renamable $x[[REGADDR]] :: (load 4)
; 64BIT-DAG:   renamable $x[[REG2:[0-9]+]] = LBZ8 4, renamable $x[[REGADDR]] :: (load 1)
; 64BIT-DAG:   renamable $x3 = RLWINM8 killed renamable $x[[REG2]], 24, 0, 7
; 64BIT-DAG:   renamable $x3 = RLDIMI killed renamable $x3, killed renamable $x[[REG1]], 32, 0
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_5Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM64PWR4:       stdu 1, -112(1)
; ASM64PWR4-NEXT:  ld [[REGADDR:[0-9]+]], LC{{[0-9]+}}(2)
; ASM64PWR4-DAG:   lwz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM64PWR4-DAG:   lbz [[REG2:[0-9]+]], 4([[REGADDR]])
; ASM64PWR4-DAG:   rlwinm 3, [[REG2]], 24, 0, 7
; ASM64PWR4-DAG:   rldimi 3, [[REG1]], 32, 0
; ASM64PWR4-NEXT:  bl .test_byval_5Byte
; ASM64PWR4-NEXT:  nop

%struct.S6 = type { [6 x i8] }

@gS6 = external global %struct.S6, align 1

define void @call_test_byval_6Byte() {
entry:
  call void @test_byval_6Byte(%struct.S6* byval(%struct.S6) align 1 @gS6)
  ret void
}

declare void @test_byval_6Byte(%struct.S6* byval(%struct.S6) align 1)

; CHECK-LABEL: name: call_test_byval_6Byte{{.*}}

; CHECKASM-LABEL: .call_test_byval_6Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS6, $x2 :: (load 8 from got)
; 64BIT-DAG:   renamable $x[[REG1:[0-9]+]] = LWZ8 0, killed renamable $x[[REGADDR]] :: (load 4)
; 64BIT-DAG:   renamable $x[[REG2:[0-9]+]] = LHZ8 4, renamable $x[[REGADDR]] :: (load 2)
; 64BIT-DAG:   renamable $x3 = RLWINM8 killed renamable $x[[REG2]], 16, 0, 15
; 64BIT-DAG:   renamable $x3 = RLDIMI killed renamable $x3, killed renamable $x[[REG1]], 32, 0
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_6Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM64PWR4:       stdu 1, -112(1)
; ASM64PWR4-NEXT:  ld [[REGADDR:[0-9]+]], LC{{[0-9]+}}(2)
; ASM64PWR4-DAG:   lwz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM64PWR4-DAG:   lhz [[REG2:[0-9]+]], 4([[REGADDR]])
; ASM64PWR4-DAG:   rlwinm 3, [[REG2]], 16, 0, 15
; ASM64PWR4-DAG:   rldimi 3, [[REG1]], 32, 0
; ASM64PWR4-NEXT:  bl .test_byval_6Byte
; ASM64PWR4-NEXT:  nop

%struct.S7 = type { [7 x i8] }

@gS7 = external global %struct.S7, align 1

define void @call_test_byval_7Byte() {
entry:
  call void @test_byval_7Byte(%struct.S7* byval(%struct.S7) align 1 @gS7)
  ret void
}

declare void @test_byval_7Byte(%struct.S7* byval(%struct.S7) align 1)

; CHECK-LABEL: name: call_test_byval_7Byte{{.*}}

; CHECKASM-LABEL: .call_test_byval_7Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS7, $x2 :: (load 8 from got)
; 64BIT-DAG:   renamable $x[[REG1:[0-9]+]] = LWZ8 0, killed renamable $x[[REGADDR]] :: (load 4)
; 64BIT-DAG:   renamable $x[[REG2:[0-9]+]] = LHZ8 4, renamable $x[[REGADDR]] :: (load 2)
; 64BIT-DAG:   renamable $x[[REG3:[0-9]+]] = LBZ8 6, renamable $x[[REGADDR]] :: (load 1)
; 64BIT-DAG:   renamable $x3 = RLWINM8 killed renamable $x[[REG3]], 8, 16, 23
; 64BIT-DAG:   renamable $x3 = RLWIMI8 killed renamable $x3, killed renamable $x[[REG2]], 16, 0, 15
; 64BIT-DAG:   renamable $x3 = RLDIMI killed renamable $x3, killed renamable $x[[REG1]], 32, 0
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_7Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM64PWR4:       stdu 1, -112(1)
; ASM64PWR4-NEXT:  ld [[REGADDR:[0-9]+]], LC{{[0-9]+}}(2)
; ASM64PWR4-DAG:   lwz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM64PWR4-DAG:   lhz [[REG2:[0-9]+]], 4([[REGADDR]])
; ASM64PWR4-DAG:   lbz [[REG3:[0-9]+]], 6([[REGADDR]])
; ASM64PWR4-DAG:   rlwinm 3, [[REG3]], 8, 16, 23
; ASM64PWR4-DAG:   rlwimi 3, [[REG2]], 16, 0, 15
; ASM64PWR4-DAG:   rldimi 3, [[REG1]], 32, 0
; ASM64PWR4-NEXT:  bl .test_byval_7Byte
; ASM64PWR4-NEXT:  nop

%struct.S8 = type { [8 x i8] }

@gS8 = external global %struct.S8, align 1

define void @call_test_byval_8Byte() {
entry:
  call void @test_byval_8Byte(%struct.S8* byval(%struct.S8) align 1 @gS8)
  ret void
}

declare void @test_byval_8Byte(%struct.S8* byval(%struct.S8) align 1)

; CHECK-LABEL: name: call_test_byval_8Byte{{.*}}

; CHECKASM-LABEL: .call_test_byval_8Byte:

; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS8, $x2 :: (load 8 from got)
; 64BIT-NEXT:  renamable $x3 = LD 0, killed renamable $x[[REGADDR]] :: (load 8)
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_8Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; ASM64PWR4:       stdu 1, -112(1)
; ASM64PWR4-NEXT:  ld [[REGADDR:[0-9]+]], LC{{[0-9]+}}(2)
; ASM64PWR4-NEXT:  ld 3, 0([[REGADDR]])
; ASM64PWR4-NEXT:  bl .test_byval_8Byte
; ASM64PWR4-NEXT:  nop

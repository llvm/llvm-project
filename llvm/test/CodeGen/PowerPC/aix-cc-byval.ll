; RUN: llc -mtriple powerpc-ibm-aix-xcoff -stop-after=machine-cp -mcpu=pwr4 \
; RUN: -mattr=-altivec -verify-machineinstrs < %s | \
; RUN: FileCheck --check-prefixes=CHECK,32BIT %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=CHECKASM,ASM32 %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp -mcpu=pwr4 \
; RUN: -mattr=-altivec -verify-machineinstrs < %s | \
; RUN: FileCheck --check-prefixes=CHECK,64BIT %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=CHECKASM,ASM64 %s

%struct.S0 = type {}

%struct.S1 = type { [1 x i8] }
@gS1 = external global %struct.S1, align 1

define void @call_test_byval_1Byte() {
entry:
  %s0 = alloca %struct.S0, align 8
  call void @test_byval_1Byte(%struct.S0* byval(%struct.S0) align 1 %s0, %struct.S1* byval(%struct.S1) align 1 @gS1)
  ret void
}

declare void @test_byval_1Byte(%struct.S0* byval(%struct.S0) align 1, %struct.S1* byval(%struct.S1) align 1)

; CHECK-LABEL: name: call_test_byval_1Byte{{.*}}

; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REG:[0-9]+]] = LWZtoc @gS1, $r2 :: (load 4 from got)
; 32BIT-NEXT:  renamable $r3 = LBZ 0, killed renamable $r[[REG]] :: (load 1)
; 32BIT-NEXT:  renamable $r3 = RLWINM killed renamable $r3, 24, 0, 7
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_1Byte>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_1Byte:

; ASM32:       stwu 1, -64(1)
; ASM32-NEXT:  lwz [[REG:[0-9]+]], LC{{[0-9]+}}(2)
; ASM32-NEXT:  lbz 3, 0([[REG]])
; ASM32-NEXT:  slwi 3, 3, 24
; ASM32-NEXT:  bl .test_byval_1Byte
; ASM32-NEXT:  nop
; ASM32-NEXT:  addi 1, 1, 64

; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REG:[0-9]+]] = LDtoc @gS1, $x2 :: (load 8 from got)
; 64BIT-NEXT:  renamable $x3 = LBZ8 0, killed renamable $x[[REG]] :: (load 1)
; 64BIT-NEXT:  renamable $x3 = RLDICR killed renamable $x3, 56, 7
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_1Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; ASM64:       std 0, 16(1)
; ASM64-NEXT:  stdu 1, -128(1)
; ASM64-NEXT:  ld [[REG:[0-9]+]], LC{{[0-9]+}}(2)
; ASM64-NEXT:  lbz 3, 0([[REG]])
; ASM64-NEXT:  sldi 3, 3, 56
; ASM64-NEXT:  bl .test_byval_1Byte
; ASM64-NEXT:  nop
; ASM64-NEXT:  addi 1, 1, 128

%struct.S2 = type { [2 x i8] }

@gS2 = external global %struct.S2, align 1

define void @call_test_byval_2Byte() {
entry:
  call void @test_byval_2Byte(%struct.S2* byval(%struct.S2) align 1 @gS2)
  ret void
}

declare void @test_byval_2Byte(%struct.S2* byval(%struct.S2) align 1)

; CHECK-LABEL: name: call_test_byval_2Byte{{.*}}

; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REG:[0-9]+]] = LWZtoc @gS2, $r2 :: (load 4 from got)
; 32BIT-NEXT:  renamable $r3 = LHZ 0, killed renamable $r[[REG]] :: (load 2)
; 32BIT-NEXT:  renamable $r3 = RLWINM killed renamable $r3, 16, 0, 15
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_2Byte>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_2Byte:

; ASM32:       stwu 1, -64(1)
; ASM32-NEXT:  lwz [[REG:[0-9]+]], LC{{[0-9]+}}(2)
; ASM32-NEXT:  lhz 3, 0([[REG]])
; ASM32-NEXT:  slwi 3, 3, 16
; ASM32-NEXT:  bl .test_byval_2Byte
; ASM32-NEXT:  nop
; ASM32-NEXT:  addi 1, 1, 64

; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REG:[0-9]+]] = LDtoc @gS2, $x2 :: (load 8 from got)
; 64BIT-NEXT:  renamable $x3 = LHZ8 0, killed renamable $x[[REG]] :: (load 2)
; 64BIT-NEXT:  renamable $x3 = RLDICR killed renamable $x3, 48, 15
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_2Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; ASM64:       std 0, 16(1)
; ASM64-NEXT:  stdu 1, -112(1)
; ASM64-NEXT:  ld [[REG:[0-9]+]], LC{{[0-9]+}}(2)
; ASM64-NEXT:  lhz 3, 0([[REG]])
; ASM64-NEXT:  sldi 3, 3, 48
; ASM64-NEXT:  bl .test_byval_2Byte
; ASM64-NEXT:  nop
; ASM64-NEXT:  addi 1, 1, 112

%struct.S3 = type { [3 x i8] }

@gS3 = external global %struct.S3, align 1

define void @call_test_byval_3Byte() {
entry:
  call void @test_byval_3Byte(%struct.S3* byval(%struct.S3) align 1 @gS3)
  ret void
}

declare void @test_byval_3Byte(%struct.S3* byval(%struct.S3) align 1)

; CHECK-LABEL: name: call_test_byval_3Byte{{.*}}

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REGADDR:[0-9]+]] = LWZtoc @gS3, $r2 :: (load 4 from got)
; 32BIT-DAG:   renamable $r[[REG1:[0-9]+]] = LHZ 0, killed renamable $r[[REGADDR]] :: (load 2)
; 32BIT-DAG:   renamable $r[[REG2:[0-9]+]] = LBZ 2, renamable $r[[REGADDR]] :: (load 1)
; 32BIT-DAG:   renamable $r3 = RLWINM killed renamable $r[[REG2]], 8, 16, 23
; 32BIT-DAG:   renamable $r3 = RLWIMI killed renamable $r3, killed renamable $r[[REG1]], 16, 0, 15
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_3Byte>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_3Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM32:       stwu 1, -64(1)
; ASM32-NEXT:  lwz [[REGADDR:[0-9]+]], LC{{[0-9]+}}(2)
; ASM32-DAG:   lhz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM32-DAG:   lbz [[REG2:[0-9]+]], 2([[REGADDR]])
; ASM32-DAG:   rlwinm 3, [[REG2]], 8, 16, 23
; ASM32-DAG:   rlwimi 3, [[REG1]], 16, 0, 15
; ASM32-NEXT:  bl .test_byval_3Byte
; ASM32-NEXT:  nop

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-DAG:   renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS3, $x2 :: (load 8 from got)
; 64BIT-DAG:   renamable $x[[REG1:[0-9]+]] = LHZ8 0, killed renamable $x[[REGADDR]] :: (load 2)
; 64BIT-DAG:   renamable $x[[REG2:[0-9]+]] = LBZ8 2, renamable $x[[REGADDR]] :: (load 1)
; 64BIT-DAG:   renamable $x3 = RLDIC killed renamable $x[[REG2]], 40, 16
; 64BIT-DAG:   renamable $x3 = RLDIMI killed renamable $x3, killed renamable $x[[REG1]], 48, 0
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_3Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM64:       stdu 1, -112(1)
; ASM64-NEXT:  ld [[REGADDR:[0-9]+]], LC{{[0-9]+}}(2)
; ASM64-DAG:   lhz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM64-DAG:   lbz [[REG2:[0-9]+]], 2([[REGADDR]])
; ASM64-DAG:   rldic 3, [[REG2]], 40, 16
; ASM64-DAG:   rldimi 3, [[REG1]], 48, 0
; ASM64-NEXT:  bl .test_byval_3Byte
; ASM64-NEXT:  nop

%struct.S4 = type { [4 x i8] }

@gS4 = external global %struct.S4, align 1

define void @call_test_byval_4Byte() {
entry:
  %s0 = alloca %struct.S0, align 8
  call void @test_byval_4Byte(%struct.S4* byval(%struct.S4) align 1 @gS4, %struct.S0* byval(%struct.S0) align 1 %s0)
  ret void
}

declare void @test_byval_4Byte(%struct.S4* byval(%struct.S4) align 1, %struct.S0* byval(%struct.S0) align 1)

; CHECK-LABEL: name: call_test_byval_4Byte{{.*}}

; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REG:[0-9]+]] = LWZtoc @gS4, $r2 :: (load 4 from got)
; 32BIT-NEXT:  renamable $r3 = LWZ 0, killed renamable $r[[REG]] :: (load 4)
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_4Byte>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_4Byte:

; ASM32:       stwu 1, -64(1)
; ASM32-NEXT:  lwz [[REG:[0-9]+]], LC{{[0-9]+}}(2)
; ASM32-NEXT:  lwz 3, 0([[REG]])
; ASM32-NEXT:  bl .test_byval_4Byte
; ASM32-NEXT:  nop
; ASM32-NEXT:  addi 1, 1, 64

; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REG:[0-9]+]] = LDtoc @gS4, $x2 :: (load 8 from got)
; 64BIT-NEXT:  renamable $x3 = LWZ8 0, killed renamable $x[[REG]] :: (load 4)
; 64BIT-NEXT:  renamable $x3 = RLDICR killed renamable $x3, 32, 31
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_4Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; ASM64:       stdu 1, -128(1)
; ASM64-NEXT:  ld [[REG:[0-9]+]], LC{{[0-9]+}}(2)
; ASM64-NEXT:  lwz 3, 0([[REG]])
; ASM64-NEXT:  sldi 3, 3, 32
; ASM64-NEXT:  bl .test_byval_4Byte
; ASM64-NEXT:  nop
; ASM64-NEXT:  addi 1, 1, 128

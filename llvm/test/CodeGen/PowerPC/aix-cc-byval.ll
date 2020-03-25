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
  %call = call zeroext i8 @test_byval_1Byte(%struct.S0* byval(%struct.S0) align 1 %s0, %struct.S1* byval(%struct.S1) align 1 @gS1)
  ret void
}


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


define zeroext i8 @test_byval_1Byte(%struct.S0* byval(%struct.S0) align 1 %s0, %struct.S1* byval(%struct.S1) align 1 %s) {
entry:
  %arrayidx = getelementptr inbounds %struct.S1, %struct.S1* %s, i32 0, i32 0, i32 0
  %0 = load i8, i8* %arrayidx, align 1
  ret i8 %0
}

; CHECK-LABEL: name:            test_byval_1Byte

; 32BIT:       fixedStack:
; 32BIT-NEXT:    - { id: 0, type: default, offset: 24, size: 4, alignment: 8, stack-id: default,
; 32BIT-NEXT:        isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,
; 32BIT:         - { id: 1, type: default, offset: 24, size: 4, alignment: 8, stack-id: default,
; 32BIT-NEXT:        isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,

; 32BIT:       bb.0.entry:
; 32BIT-NEXT:    liveins: $r3
; 32BIT:         STW killed renamable $r3, 0, %fixed-stack.0 :: (store 4 into %fixed-stack.0, align 8)
; 32BIT-NEXT:    renamable $r3 = LBZ 0,  %fixed-stack.0 :: (dereferenceable load 1
; 32BIT-NEXT:    BLR

; 64BIT:       fixedStack:
; 64BIT-NEXT:    - { id: 0, type: default, offset: 48, size: 8, alignment: 16, stack-id: default,
; 64BIT-NEXT:        isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,
; 64BIT:         - { id: 1, type: default, offset: 48, size: 8, alignment: 16, stack-id: default,
; 64BIT-NEXT:        isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,

; 64BIT:      bb.0.entry:
; 64BIT-NEXT:   liveins: $x3
; 64BIT:        STD killed renamable $x3, 0, %fixed-stack.0 :: (store 8 into %fixed-stack.0, align 16)
; 64BIT-NEXT:   renamable $x3 = LBZ8 0, %fixed-stack.0 :: (dereferenceable load 1

; CHECKASM-LABEL: .test_byval_1Byte:

; ASM32:        stw 3, 24(1)
; ASM32-NEXT:   lbz 3, 24(1)
; ASM32-NEXT:   blr

; ASM64:        std 3, 48(1)
; ASM64-NEXT:   lbz 3, 48(1)
; ASM64-NEXT:   blr

%struct.S2 = type { [2 x i8] }

@gS2 = external global %struct.S2, align 1

define void @call_test_byval_2Byte() {
entry:
  %call = call zeroext i8 @test_byval_2Byte(%struct.S2* byval(%struct.S2) align 1 @gS2)
  ret void
}

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


define zeroext i8 @test_byval_2Byte(%struct.S2* byval(%struct.S2) align 1 %s) {
entry:
  %arrayidx = getelementptr inbounds %struct.S2, %struct.S2* %s, i32 0, i32 0, i32 1
  %0 = load i8, i8* %arrayidx, align 1
  ret i8 %0
}

; CHECK-LABEL: name:            test_byval_2Byte
; 32BIT:      fixedStack:
; 32BIT-NEXT:   - { id: 0, type: default, offset: 24, size: 4, alignment: 8, stack-id: default,

; 32BIT:      bb.0.entry:
; 32BIT-NEXT:   liveins: $r3
; 32BIT:        STW killed renamable $r3, 0, %fixed-stack.0 :: (store 4 into %fixed-stack.0, align 8)
; 32BIT-NEXT:   renamable $r3 = LBZ 1, %fixed-stack.0 :: (dereferenceable load 1

; 64BIT:      fixedStack:
; 64BIT-NEXT:   - { id: 0, type: default, offset: 48, size: 8, alignment: 16, stack-id: default,

; 64BIT:      bb.0.entry:
; 64BIT-NEXT:   liveins: $x3
; 64BIT:        STD killed renamable $x3, 0, %fixed-stack.0 :: (store 8 into %fixed-stack.0, align 16)
; 64BIT-NEXT:   renamable $x3 = LBZ8 1, %fixed-stack.0 :: (dereferenceable load 1

; CHECKASM-LABEL: .test_byval_2Byte:

; ASM32:        stw 3, 24(1)
; ASM32-NEXT:   lbz 3, 25(1)
; ASM32-NEXT:   blr

; ASM64:        std 3, 48(1)
; ASM64-NEXT:   lbz 3, 49(1)
; ASM64-NEXT:   blr


%struct.S3 = type <{ i8, i16 }>
@gS3 = external global %struct.S3, align 1

define void @call_test_byval_3Byte() {
entry:
  %call = call zeroext i16 @test_byval_3Byte(%struct.S3* byval(%struct.S3) align 1 @gS3)
  ret void
}

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


define zeroext i16 @test_byval_3Byte(%struct.S3* byval(%struct.S3) align 1 %s) {
entry:
  %gep = getelementptr inbounds %struct.S3, %struct.S3* %s, i32 0, i32 1
  %0 = load i16, i16* %gep, align 1
  ret i16 %0
}

; CHECK-LABEL: name:            test_byval_3Byte

; 32BIT:      fixedStack:
; 32BIT-NEXT:   - { id: 0, type: default, offset: 24, size: 4, alignment: 8, stack-id: default,

; 32BIT:      bb.0.entry:
; 32BIT-NEXT:   liveins: $r3
; 32BIT:        STW killed renamable $r3, 0, %fixed-stack.0 :: (store 4 into %fixed-stack.0, align 8)
; 32BIT-NEXT:   renamable $r3 = LHZ 1, %fixed-stack.0 :: (dereferenceable load 2

; 64BIT:      fixedStack:
; 64BIT-NEXT:   - { id: 0, type: default, offset: 48, size: 8, alignment: 16, stack-id: default,

; 64BIT:      bb.0.entry:
; 64BIT-NEXT:   liveins: $x3
; 64BIT:        STD killed renamable $x3, 0, %fixed-stack.0 :: (store 8 into %fixed-stack.0, align 16)
; 64BIT-NEXT:   renamable $x3 = LHZ8 1, %fixed-stack.0 :: (dereferenceable load 2


; CHECKASM-LABEL: .test_byval_3Byte:

; ASM32:        stw 3, 24(1)
; ASM32-NEXT:   lhz 3, 25(1)
; ASM32-NEXT:   blr

; ASM64:        std 3, 48(1)
; ASM64-NEXT:   lhz 3, 49(1)
; ASM64-NEXT:   blr


%struct.S4 = type { [4 x i8] }
%struct.S4A = type { i32 }

@gS4 = external global %struct.S4, align 1

define void @call_test_byval_4Byte() {
entry:
  %s0 = alloca %struct.S0, align 8
  %s4a = alloca %struct.S4A, align 4
  %call = call signext i32 @test_byval_4Byte(%struct.S4* byval(%struct.S4) align 1 @gS4, %struct.S0* byval(%struct.S0) align 1 %s0, %struct.S4A* byval(%struct.S4A) align 4 %s4a)
  ret void
}

; CHECK-LABEL: name: call_test_byval_4Byte{{.*}}

; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REG:[0-9]+]] = LWZtoc @gS4, $r2 :: (load 4 from got)
; 32BIT-DAG:   renamable $r3 = LWZ 0, killed renamable $r[[REG]] :: (load 4)
; 32BIT-DAG:   renamable $r4 = LWZ 0, %stack.1.s4a :: (load 4 from %stack.1.s4a, align 8)
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_4Byte>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3,  implicit $r4, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_4Byte:

; ASM32:       stwu 1, -80(1)
; ASM32-NEXT:  lwz [[REG:[0-9]+]], LC{{[0-9]+}}(2)
; ASM32-DAG:   lwz 3, 0([[REG]])
; ASM32-DAG:   lwz 4, 64(1)
; ASM32-NEXT:  bl .test_byval_4Byte
; ASM32-NEXT:  nop
; ASM32-NEXT:  addi 1, 1, 80

; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS4, $x2 :: (load 8 from got)
; 64BIT-DAG:   renamable $x[[LD1:[0-9]+]] = LWZ8 0, killed renamable $x[[REGADDR]] :: (load 4)
; 64BIT-DAG:   renamable $x[[LD2:[0-9]+]] = LWZ8 0, %stack.1.s4a :: (load 4 from %stack.1.s4a, align 8)
; 64BIT-DAG:   renamable $x3 = RLDICR killed renamable $x[[LD1]], 32, 31
; 64BIT-DAG:   renamable $x4 = RLDICR killed renamable $x[[LD2]], 32, 31
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_4Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3,  implicit $x4, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; ASM64:       stdu 1, -128(1)
; ASM64-NEXT:  ld [[REGADDR:[0-9]+]], LC{{[0-9]+}}(2)
; ASM64-DAG:   lwz [[LD1:[0-9]+]], 0([[REGADDR]])
; ASM64-DAG:   lwz [[LD2:[0-9]+]], 112(1)
; ASM64-DAG:   sldi 3, [[LD1]], 32
; ASM64-DAG:   sldi 4, [[LD2]], 32
; ASM64-NEXT:  bl .test_byval_4Byte
; ASM64-NEXT:  nop
; ASM64-NEXT:  addi 1, 1, 128


define signext i32 @test_byval_4Byte(%struct.S4* byval(%struct.S4) align 1 %s, %struct.S0* byval(%struct.S0) align 1, %struct.S4A* byval(%struct.S4A) align 4 %s4a) {
entry:
  %arrayidx = getelementptr inbounds %struct.S4, %struct.S4* %s, i32 0, i32 0, i32 3
  %gep = getelementptr inbounds %struct.S4A, %struct.S4A* %s4a, i32 0, i32 0
  %1 = load i8, i8* %arrayidx, align 1
  %2 = load i32, i32* %gep, align 4
  %conv = zext i8 %1 to i32
  %add = add nsw i32 %2, %conv
  ret i32 %add
}

; CHECK-LABEL: name:            test_byval_4Byte

; 32BIT:      fixedStack:
; 32BIT-NEXT:   - { id: 0, type: default, offset: 28, size: 4, alignment: 4, stack-id: default,
; 32BIT-NEXT:       isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,
; 32BIT:        - { id: 1, type: default, offset: 28, size: 4, alignment: 4, stack-id: default,
; 32BIT-NEXT:       isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,
; 32BIT:        - { id: 2, type: default, offset: 24, size: 4, alignment: 8, stack-id: default,
; 32BIT-NEXT:       isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,

; 32BIT:      bb.0.entry:
; 32BIT-NEXT:   liveins: $r3
; 32BIT:        STW renamable $r3, 0, %fixed-stack.2 :: (store 4 into %fixed-stack.2, align 8)
; 32BIT-DAG:    STW killed renamable $r4, 0, %fixed-stack.0 :: (store 4 into %fixed-stack.0)
; 32BIT-DAG:    renamable $r[[SCRATCH:[0-9]+]] = RLWINM killed renamable $r3, 0, 24, 31
; 32BIT-DAG:    renamable $r3 = nsw ADD4 renamable $r4, killed renamable $r[[SCRATCH]]
; 32BIT:        BLR

; 64BIT:      fixedStack:
; 64BIT-NEXT: - { id: 0, type: default, offset: 56, size: 8, alignment: 8, stack-id: default,
; 64BIT-NEXT:     isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,
; 64BIT:      - { id: 1, type: default, offset: 56, size: 8, alignment: 8, stack-id: default,
; 64BIT-NEXT:     isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,
; 64BIT:      - { id: 2, type: default, offset: 48, size: 8, alignment: 16, stack-id: default,
; 64BIT-NEXT:     isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,

; 64BIT:      bb.0.entry:
; 64BIT-NEXT:   liveins: $x3
; 64BIT:        STD killed renamable $x3, 0, %fixed-stack.2 :: (store 8 into %fixed-stack.2, align 16)
; 64BIT-NEXT:   STD killed renamable $x4, 0, %fixed-stack.0 :: (store 8 into %fixed-stack.0)
; 64BIT-DAG:    renamable $r[[SCRATCH1:[0-9]+]] = LBZ 3, %fixed-stack.2 :: (dereferenceable load 1
; 64BIT-DAG:    renamable $r[[SCRATCH2:[0-9]+]] = LWZ 0, %fixed-stack.0 :: (dereferenceable load 4
; 64BIT-NEXT:   renamable $r[[SCRATCH3:[0-9]+]] = nsw ADD4 killed renamable $r[[SCRATCH2]], killed renamable $r[[SCRATCH1]]
; 64BIT-NEXT:   renamable $x3 = EXTSW_32_64 killed renamable $r[[SCRATCH3]]
; 64BIT-NEXT:   BLR8

; CHECKASM-LABEL: .test_byval_4Byte:

; ASM32:        stw 3, 24(1)
; ASM32-DAG:    stw 4, 28(1)
; ASM32-DAG:    clrlwi  [[SCRATCH:[0-9]+]], 3, 24
; ASM32-DAG:    add 3, 4, [[SCRATCH]]
; ASM32-NEXT:   blr

; ASM64:        std 3, 48(1)
; ASM64-NEXT:   std 4, 56(1)
; ASM64-DAG:    lbz [[SCRATCH1:[0-9]+]], 51(1)
; ASM64-DAG:    lwz [[SCRATCH2:[0-9]+]], 56(1)
; ASM64-NEXT:   add [[SCRATCH3:[0-9]+]], [[SCRATCH2]], [[SCRATCH1]]
; ASM64-NEXT:   extsw 3, [[SCRATCH3]]
; ASM64-NEXT:   blr

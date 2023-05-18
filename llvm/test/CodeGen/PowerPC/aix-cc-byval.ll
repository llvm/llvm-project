; RUN: llc -mtriple powerpc-ibm-aix-xcoff -stop-after=machine-cp -mcpu=pwr4 \
; RUN:  -mattr=-altivec -verify-machineinstrs < %s | \
; RUN: FileCheck --check-prefixes=CHECK,32BIT %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=CHECKASM,ASM32 %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp -mcpu=pwr4 \
; RUN:  -mattr=-altivec -verify-machineinstrs < %s | \
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
  %call = call zeroext i8 @test_byval_1Byte(ptr byval(%struct.S0) align 1 %s0, ptr byval(%struct.S1) align 1 @gS1)
  ret void
}


; CHECK-LABEL: name: call_test_byval_1Byte{{.*}}

; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REG:[0-9]+]] = LWZtoc @gS1, $r2 :: (load (s32) from got)
; 32BIT-NEXT:  renamable $r3 = LBZ 0, killed renamable $r[[REG]] :: (load (s8))
; 32BIT-NEXT:  renamable $r3 = RLWINM killed renamable $r3, 24, 0, 7
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_1Byte>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_1Byte:

; ASM32:       stwu 1, -64(1)
; ASM32-NEXT:  lwz [[REG:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32-NEXT:  stw 0, 72(1)
; ASM32-NEXT:  lbz 3, 0([[REG]])
; ASM32-NEXT:  slwi 3, 3, 24
; ASM32-NEXT:  bl .test_byval_1Byte
; ASM32-NEXT:  nop
; ASM32-NEXT:  addi 1, 1, 64

; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REG:[0-9]+]] = LDtoc @gS1, $x2 :: (load (s64) from got)
; 64BIT-NEXT:  renamable $x3 = LBZ8 0, killed renamable $x[[REG]] :: (load (s8))
; 64BIT-NEXT:  renamable $x3 = RLDICR killed renamable $x3, 56, 7
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_1Byte>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; ASM64:       stdu 1, -128(1)
; ASM64-NEXT:  ld [[REG:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64-NEXT:  std 0, 144(1)
; ASM64-NEXT:  lbz 3, 0([[REG]])
; ASM64-NEXT:  sldi 3, 3, 56
; ASM64-NEXT:  bl .test_byval_1Byte
; ASM64-NEXT:  nop
; ASM64-NEXT:  addi 1, 1, 128


define zeroext i8 @test_byval_1Byte(ptr byval(%struct.S0) align 1 %s0, ptr byval(%struct.S1) align 1 %s) {
entry:
  %0 = load i8, ptr %s, align 1
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
; 32BIT:         renamable $r4 = COPY $r3
; 32BIT:         renamable $r3 = RLWINM $r3, 8, 24, 31
; 32BIT:         STW killed renamable $r4, 0, %fixed-stack.0 :: (store (s32) into %fixed-stack.0, align 8)
; 32BIT-NEXT:    BLR

; 64BIT:       fixedStack:
; 64BIT-NEXT:    - { id: 0, type: default, offset: 48, size: 8, alignment: 16, stack-id: default,
; 64BIT-NEXT:        isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,
; 64BIT:         - { id: 1, type: default, offset: 48, size: 8, alignment: 16, stack-id: default,
; 64BIT-NEXT:        isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,

; 64BIT:      bb.0.entry:
; 64BIT-NEXT:   liveins: $x3
; 64BIT:        renamable $x4 = COPY $x3
; 64BIT:        renamable $x3 = RLDICL $x3, 8, 56
; 64BIT:        STD killed renamable $x4, 0, %fixed-stack.0 :: (store (s64) into %fixed-stack.0, align 16)

; CHECKASM-LABEL: .test_byval_1Byte:

; ASM32:      mr 4, 3
; ASM32-NEXT: srwi 3, 3, 24
; ASM32-NEXT: stw 4, 24(1)
; ASM32-NEXT: blr

; ASM64:      mr 4, 3
; ASM64-NEXT: rldicl 3, 3, 8, 56
; ASM64-NEXT: std 4, 48(1)
; ASM64-NEXT: blr


@f = common global float 0.000000e+00, align 4

%struct.S2 = type { [2 x i8] }

@gS2 = external global %struct.S2, align 1

define void @call_test_byval_2Byte() {
entry:
  %0 = load float, ptr @f, align 4
  %call = call zeroext i8 @test_byval_2Byte(i32 signext 42, float %0, ptr byval(%struct.S2) align 1 @gS2, float %0, i32 signext 43)
  ret void
}

; CHECK-LABEL: name: call_test_byval_2Byte{{.*}}

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 32BIT:       renamable $r[[REG1:[0-9]+]] = LWZtoc @f, $r2 :: (load (s32) from got)
; 32BIT-NEXT:  renamable $f1 = LFS 0, killed renamable $r[[REG1]] :: (dereferenceable load (s32) from @f)
; 32BIT-NEXT:  ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-DAG:   $r3 = LI 42
; 32BIT-DAG:   renamable $r[[REG2:[0-9]+]] = LWZtoc @gS2, $r2 :: (load (s32) from got)
; 32BIT-DAG:   renamable $r[[REG3:[0-9]+]] = LHZ 0, killed renamable $r[[REG2]] :: (load (s16))
; 32BIT-DAG:   renamable $r5 = RLWINM killed renamable $r[[REG3]], 16, 0, 15
; 32BIT-DAG:   $f2 = COPY renamable $f1
; 32BIT-DAG:   $r7 = LI 43
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_2Byte>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $f1, implicit $r5, implicit killed $f2, implicit killed $r7, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_2Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM32:       stwu 1, -64(1)
; ASM32-DAG:   li 3, 42
; ASM32-DAG:   lwz [[REG1:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32-DAG:   lfs 1, 0([[REG1]])
; ASM32-DAG:   lwz [[REG2:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32-DAG:   lhz [[REG3:[0-9]+]], 0([[REG2]])
; ASM32-DAG:   slwi 5, [[REG3]], 16
; ASM32-DAG:   fmr 2, 1
; ASM32-DAG:   li 7, 43
; ASM32-NEXT:  bl .test_byval_2Byte
; ASM32-NEXT:  nop
; ASM32-NEXT:  addi 1, 1, 64

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 64BIT:       renamable $x[[REG1:[0-9]+]] = LDtoc @f, $x2 :: (load (s64) from got)
; 64BIT-NEXT:  renamable $f1 = LFS 0, killed renamable $x[[REG1]] :: (dereferenceable load (s32) from @f)
; 64BIT-NEXT:  ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-DAG:   $x3 = LI8 42
; 64BIT-DAG:   renamable $x[[REG2:[0-9]+]] = LDtoc @gS2, $x2 :: (load (s64) from got)
; 64BIT-DAG:   renamable $x[[REG3:[0-9]+]] = LHZ8 0, killed renamable $x[[REG2]] :: (load (s16))
; 64BIT-DAG:   renamable $x5 = RLDICR killed renamable $x[[REG3]], 48, 15
; 64BIT-DAG:   $f2 = COPY renamable $f1
; 64BIT-DAG:   $x7 = LI8 43
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_2Byte>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $f1, implicit $x5, implicit killed $f2, implicit killed $x7, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM64:       stdu 1, -112(1)
; ASM64-DAG:   std 0, 128(1)
; ASM64-DAG:   li 3, 42
; ASM64-DAG:   ld [[REG1:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64-DAG:   lfs 1, 0([[REG1]])
; ASM64-DAG:   ld [[REG2:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64-DAG:   lhz [[REG3:[0-9]+]], 0([[REG2]])
; ASM64-DAG:   sldi 5, [[REG3]], 48
; ASM64-DAG:   fmr 2, 1
; ASM64-DAG:   li 7, 43
; ASM64-NEXT:  bl .test_byval_2Byte
; ASM64-NEXT:  nop
; ASM64-NEXT:  addi 1, 1, 112

define zeroext i8 @test_byval_2Byte(i32, float, ptr byval(%struct.S2) align 1 %s, float, i32) {
entry:
  %arrayidx = getelementptr inbounds %struct.S2, ptr %s, i32 0, i32 0, i32 1
  %4 = load i8, ptr %arrayidx, align 1
  ret i8 %4
}

; CHECK-LABEL: name:            test_byval_2Byte
; 32BIT:      fixedStack:
; 32BIT-NEXT:   - { id: 0, type: default, offset: 32, size: 4, alignment: 16, stack-id: default,

; 32BIT:      bb.0.entry:
; 32BIT-NEXT:   liveins: $r5
; 32BIT:        STW killed renamable $r5, 0, %fixed-stack.0 :: (store (s32) into %fixed-stack.0, align 16)
; 32BIT-NEXT:   renamable $r3 = LBZ 1, %fixed-stack.0 :: (dereferenceable load (s8)

; 64BIT:      fixedStack:
; 64BIT-NEXT:   - { id: 0, type: default, offset: 64, size: 8, alignment: 16, stack-id: default,

; 64BIT:      bb.0.entry:
; 64BIT-NEXT:   liveins: $x5
; 64BIT:        STD killed renamable $x5, 0, %fixed-stack.0 :: (store (s64) into %fixed-stack.0, align 16)
; 64BIT-NEXT:   renamable $x3 = LBZ8 1, %fixed-stack.0 :: (dereferenceable load (s8)

; CHECKASM-LABEL: .test_byval_2Byte:

; ASM32:        stw 5, 32(1)
; ASM32-NEXT:   lbz 3, 33(1)
; ASM32-NEXT:   blr

; ASM64:        std 5, 64(1)
; ASM64-NEXT:   lbz 3, 65(1)
; ASM64-NEXT:   blr


%struct.S3 = type <{ i8, i16 }>
@gS3 = external global %struct.S3, align 1

define void @call_test_byval_3Byte() {
entry:
  %call = call zeroext i16 @test_byval_3Byte(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, ptr byval(%struct.S3) align 1 @gS3, i32 42)
  ret void
}

; CHECK-LABEL: name: call_test_byval_3Byte{{.*}}

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 32BIT:       ADJCALLSTACKDOWN 60, 0, implicit-def dead $r1, implicit $r1
; 32BIT-DAG:   $r3 = LI 1
; 32BIT-DAG:   $r4 = LI 2
; 32BIT-DAG:   $r5 = LI 3
; 32BIT-DAG:   $r6 = LI 4
; 32BIT-DAG:   $r7 = LI 5
; 32BIT-DAG:   $r8 = LI 6
; 32BIT-DAG:   $r9 = LI 7
; 32BIT-DAG:   renamable $r[[REGADDR:[0-9]+]] = LWZtoc @gS3, $r2 :: (load (s32) from got)
; 32BIT-DAG:   renamable $r[[REG1:[0-9]+]] = LHZ 0, killed renamable $r[[REGADDR]] :: (load (s16))
; 32BIT-DAG:   renamable $r[[REG2:[0-9]+]] = LBZ 2, renamable $r[[REGADDR]] :: (load (s8))
; 32BIT-DAG:   renamable $r10 = RLWINM killed renamable $r[[REG2]], 8, 16, 23
; 32BIT-DAG:   renamable $r10 = RLWIMI killed renamable $r10, killed renamable $r[[REG1]], 16, 0, 15
; 32BIT-DAG:   renamable $r[[REG3:[0-9]+]] = LI 42
; 32BIT-DAG:   STW killed renamable $r[[REG3]], 56, $r1 :: (store (s32))
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_3Byte>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit killed $r5, implicit killed $r6, implicit killed $r7, implicit killed $r8, implicit killed $r9, implicit $r10, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 60, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_3Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM32:       stwu 1, -64(1)
; ASM32-DAG:   li 3, 1
; ASM32-DAG:   li 4, 2
; ASM32-DAG:   li 5, 3
; ASM32-DAG:   li 6, 4
; ASM32-DAG:   li 7, 5
; ASM32-DAG:   li 8, 6
; ASM32-DAG:   li 9, 7
; ASM32-DAG:   lwz [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32-DAG:   lhz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM32-DAG:   lbz [[REG2:[0-9]+]], 2([[REGADDR]])
; ASM32-DAG:   rlwinm 10, [[REG2]], 8, 16, 23
; ASM32-DAG:   rlwimi 10, [[REG1]], 16, 0, 15
; ASM32-DAG:   li [[REG3:[0-9]+]], 42
; ASM32-DAG:   stw [[REG3]], 56(1)
; ASM32-NEXT:  bl .test_byval_3Byte
; ASM32-NEXT:  nop

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 64BIT:       ADJCALLSTACKDOWN 120, 0, implicit-def dead $r1, implicit $r1
; 64BIT-DAG:   $x3 = LI8 1
; 64BIT-DAG:   $x4 = LI8 2
; 64BIT-DAG:   $x5 = LI8 3
; 64BIT-DAG:   $x6 = LI8 4
; 64BIT-DAG:   $x7 = LI8 5
; 64BIT-DAG:   $x8 = LI8 6
; 64BIT-DAG:   $x9 = LI8 7
; 64BIT-DAG:   renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS3, $x2 :: (load (s64) from got)
; 64BIT-DAG:   renamable $x[[REG1:[0-9]+]] = LHZ8 0, killed renamable $x[[REGADDR]] :: (load (s16))
; 64BIT-DAG:   renamable $x[[REG2:[0-9]+]] = LBZ8 2, renamable $x[[REGADDR]] :: (load (s8))
; 64BIT-DAG:   renamable $x10 = RLDIC killed renamable $x[[REG2]], 40, 16
; 64BIT-DAG:   renamable $x10 = RLDIMI killed renamable $x10, killed renamable $x[[REG1]], 48, 0
; 64BIT-DAG:   $x[[REG3:[0-9]+]] = LI8 42
; 64BIT-DAG:   STD killed renamable $x[[REG3]], 112, $x1 :: (store (s64))
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_3Byte>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x4, implicit killed $x5, implicit killed $x6, implicit killed $x7, implicit killed $x8, implicit killed $x9, implicit $x10, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 120, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM64:       stdu 1, -128(1)
; ASM64-DAG:   li 3, 1
; ASM64-DAG:   li 4, 2
; ASM64-DAG:   li 5, 3
; ASM64-DAG:   li 6, 4
; ASM64-DAG:   li 7, 5
; ASM64-DAG:   li 8, 6
; ASM64-DAG:   li 9, 7
; ASM64-DAG:   ld [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64-DAG:   lhz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM64-DAG:   lbz [[REG2:[0-9]+]], 2([[REGADDR]])
; ASM64-DAG:   rldic 10, [[REG2]], 40, 16
; ASM64-DAG:   rldimi 10, [[REG1]], 48, 0
; ASM64-DAG:   li [[REG3:[0-9]+]], 42
; ASM64-DAG:   std [[REG3]], 112(1)
; ASM64-NEXT:  bl .test_byval_3Byte
; ASM64-NEXT:  nop


define zeroext i16 @test_byval_3Byte(i32, i32, i32, i32, i32, i32, i32, ptr byval(%struct.S3) align 1 %s, i32) {
entry:
  %gep = getelementptr inbounds %struct.S3, ptr %s, i32 0, i32 1
  %8 = load i16, ptr %gep, align 1
  ret i16 %8
}

; CHECK-LABEL: name:            test_byval_3Byte

; 32BIT:       fixedStack:
; 32BIT-NEXT:    - { id: 0, type: default, offset: 56, size: 4, alignment: 8, stack-id: default,
; 32BIT:         - { id: 1, type: default, offset: 52, size: 4, alignment: 4, stack-id: default,

; 32BIT-LABEL: bb.0.entry:
; 32BIT-NEXT:    liveins: $r10
; 32BIT:         STW killed renamable $r10, 0, %fixed-stack.1 :: (store (s32) into %fixed-stack.1)
; 32BIT-NEXT:    renamable $r3 = LHZ 1, %fixed-stack.1 :: (dereferenceable load (s16)

; 64BIT:       fixedStack:
; 64BIT-NEXT:     - { id: 0, type: default, offset: 116, size: 4, alignment: 4, stack-id: default,
; 64BIT:          - { id: 1, type: default, offset: 104, size: 8, alignment: 8, stack-id: default,

; 64BIT-LABEL: bb.0.entry:
; 64BIT-NEXT:    liveins: $x10
; 64BIT:         STD killed renamable $x10, 0, %fixed-stack.1 :: (store (s64) into %fixed-stack.1)
; 64BIT-NEXT:    renamable $x3 = LHZ8 1, %fixed-stack.1 :: (dereferenceable load (s16)

; CHECKASM-LABEL: .test_byval_3Byte:

; ASM32:        stw 10, 52(1)
; ASM32-NEXT:   lhz 3, 53(1)
; ASM32-NEXT:   blr

; ASM64:        std 10, 104(1)
; ASM64-NEXT:   lhz 3, 105(1)
; ASM64-NEXT:   blr


%struct.S4 = type { [4 x i8] }
%struct.S4A = type { i32 }

@gS4 = external global %struct.S4, align 1

define void @call_test_byval_4Byte() {
entry:
  %s0 = alloca %struct.S0, align 8
  %s4a = alloca %struct.S4A, align 8
  %call = call signext i32 @test_byval_4Byte(ptr byval(%struct.S4) align 1 @gS4, ptr byval(%struct.S0) align 1 %s0, ptr byval(%struct.S4A) align 4 %s4a)
  ret void
}

; CHECK-LABEL: name: call_test_byval_4Byte{{.*}}

; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REG:[0-9]+]] = LWZtoc @gS4, $r2 :: (load (s32) from got)
; 32BIT-DAG:   renamable $r3 = LWZ 0, killed renamable $r[[REG]] :: (load (s32))
; 32BIT-DAG:   renamable $r4 = LWZ 0, %stack.1.s4a :: (load (s32) from %stack.1.s4a, align 8)
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_4Byte>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3,  implicit $r4, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_4Byte:

; ASM32:       stwu 1, -80(1)
; ASM32-NEXT:  lwz [[REG:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32-DAG:   lwz 3, 0([[REG]])
; ASM32-DAG:   lwz 4, 64(1)
; ASM32-NEXT:  bl .test_byval_4Byte
; ASM32-NEXT:  nop
; ASM32-NEXT:  addi 1, 1, 80

; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS4, $x2 :: (load (s64) from got)
; 64BIT-DAG:   renamable $x[[LD1:[0-9]+]] = LWZ8 0, killed renamable $x[[REGADDR]] :: (load (s32))
; 64BIT-DAG:   renamable $x[[LD2:[0-9]+]] = LWZ8 0, %stack.1.s4a :: (load (s32) from %stack.1.s4a, align 8)
; 64BIT-DAG:   renamable $x3 = RLDICR killed renamable $x[[LD1]], 32, 31
; 64BIT-DAG:   renamable $x4 = RLDICR killed renamable $x[[LD2]], 32, 31
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_4Byte>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3,  implicit $x4, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; ASM64:       stdu 1, -128(1)
; ASM64-NEXT:  ld [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64-DAG:   lwz [[LD1:[0-9]+]], 0([[REGADDR]])
; ASM64-DAG:   lwz [[LD2:[0-9]+]], 112(1)
; ASM64-DAG:   sldi 3, [[LD1]], 32
; ASM64-DAG:   sldi 4, [[LD2]], 32
; ASM64-NEXT:  bl .test_byval_4Byte
; ASM64-NEXT:  nop
; ASM64-NEXT:  addi 1, 1, 128


define signext i32 @test_byval_4Byte(ptr byval(%struct.S4) align 1 %s, ptr byval(%struct.S0) align 1, ptr byval(%struct.S4A) align 4 %s4a) {
entry:
  %arrayidx = getelementptr inbounds %struct.S4, ptr %s, i32 0, i32 0, i32 3
  %1 = load i8, ptr %arrayidx, align 1
  %2 = load i32, ptr %s4a, align 4
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
; 32BIT:        STW renamable $r3, 0, %fixed-stack.2 :: (store (s32) into %fixed-stack.2, align 8)
; 32BIT-DAG:    STW killed renamable $r4, 0, %fixed-stack.0 :: (store (s32) into %fixed-stack.0)
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
; 64BIT:        STD killed renamable $x3, 0, %fixed-stack.2 :: (store (s64) into %fixed-stack.2, align 16)
; 64BIT:        STD renamable $x4, 0, %fixed-stack.0 :: (store (s64) into %fixed-stack.0)
; 64BIT-DAG:    renamable $r[[SCRATCH1:[0-9]+]] = LBZ 3, %fixed-stack.2 :: (dereferenceable load (s8)
; 64BIT-DAG:    renamable $x[[SCRATCH2:[0-9]+]] = RLDICL killed renamable $x4, 32, 32
; 64BIT-NEXT:   renamable $r[[SCRATCH3:[0-9]+]] = nsw ADD4 renamable $r[[SCRATCH2]], killed renamable $r[[SCRATCH1]], implicit killed $x[[SCRATCH2]]
; 64BIT-NEXT:   renamable $x3 = EXTSW_32_64 killed renamable $r[[SCRATCH3]]
; 64BIT-NEXT:   BLR8

; CHECKASM-LABEL: .test_byval_4Byte:

; ASM32:        stw 3, 24(1)
; ASM32-DAG:    stw 4, 28(1)
; ASM32-DAG:    clrlwi  [[SCRATCH:[0-9]+]], 3, 24
; ASM32-DAG:    add 3, 4, [[SCRATCH]]
; ASM32-NEXT:   blr

; ASM64:        std 3, 48(1)
; ASM64-NEXT:   lbz [[SCRATCH1:[0-9]+]], 51(1)
; ASM64-NEXT:   std 4, 56(1)
; ASM64-NEXT:   rldicl [[SCRATCH2:[0-9]+]], 4, 32, 32
; ASM64-NEXT:   add [[SCRATCH3:[0-9]+]], [[SCRATCH2]], [[SCRATCH1]]
; ASM64-NEXT:   extsw 3, [[SCRATCH3]]
; ASM64-NEXT:   blr


%struct.S5 = type { [5 x i8] }

@gS5 = external global %struct.S5, align 1

define void @call_test_byval_5Byte() {
entry:
  %call = call zeroext i8 @test_byval_5Byte(ptr byval(%struct.S5) align 1 @gS5)
  ret void
}

declare zeroext i8 @test_byval_5Byte(ptr byval(%struct.S5) align 1)

; CHECK-LABEL: name: call_test_byval_5Byte{{.*}}

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REGADDR:[0-9]+]] = LWZtoc @gS5, $r2 :: (load (s32) from got)
; 32BIT-DAG:   renamable $r[[REG1:[0-9]+]] = LBZ 4, renamable $r[[REGADDR]] :: (load (s8))
; 32BIT-DAG:   renamable $r3 = LWZ 0, killed renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r4 = RLWINM killed renamable $r[[REG1]], 24, 0, 7
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_5Byte[PR]>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit $r2, implicit-def $r1
; 32BIT-NEXT:   ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_5Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM32:       stwu 1, -64(1)
; ASM32-NEXT:  lwz [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32-DAG:   lbz [[REG1:[0-9]+]], 4([[REGADDR]])
; ASM32-DAG:   lwz 3, 0([[REGADDR]])
; ASM32-DAG:   slwi 4, [[REG1]], 24
; ASM32-NEXT:  bl .test_byval_5Byte[PR]
; ASM32-NEXT:  nop

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS5, $x2 :: (load (s64) from got)
; 64BIT-DAG:   renamable $x[[REG1:[0-9]+]] = LWZ8 0, killed renamable $x[[REGADDR]] :: (load (s32))
; 64BIT-DAG:   renamable $x[[REG2:[0-9]+]] = LBZ8 4, renamable $x[[REGADDR]] :: (load (s8))
; 64BIT-DAG:   renamable $x3 = RLWINM8 killed renamable $x[[REG2]], 24, 0, 7
; 64BIT-DAG:   renamable $x3 = RLDIMI killed renamable $x3, killed renamable $x[[REG1]], 32, 0
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_5Byte[PR]>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM64:       stdu 1, -112(1)
; ASM64-NEXT:  ld [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64-DAG:   lwz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM64-DAG:   lbz [[REG2:[0-9]+]], 4([[REGADDR]])
; ASM64-DAG:   rlwinm 3, [[REG2]], 24, 0, 7
; ASM64-DAG:   rldimi 3, [[REG1]], 32, 0
; ASM64-NEXT:  bl .test_byval_5Byte[PR]
; ASM64-NEXT:  nop


%struct.S6 = type { [6 x i8] }

@gS6 = external global %struct.S6, align 1

define void @call_test_byval_6Byte() {
entry:
  %call = call zeroext i8 @test_byval_6Byte(ptr byval(%struct.S6) align 1 @gS6)
  ret void
}

declare zeroext i8 @test_byval_6Byte(ptr byval(%struct.S6) align 1)

; CHECK-LABEL: name: call_test_byval_6Byte{{.*}}

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REGADDR:[0-9]+]] = LWZtoc @gS6, $r2 :: (load (s32) from got)
; 32BIT-DAG:   renamable $r[[REG1:[0-9]+]] = LHZ 4, renamable $r[[REGADDR]] :: (load (s16))
; 32BIT-DAG:   renamable $r3 = LWZ 0, killed renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r4 = RLWINM killed renamable $r[[REG1]], 16, 0, 15
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_6Byte[PR]>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_6Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM32:       stwu 1, -64(1)
; ASM32-NEXT:  lwz [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32-DAG:   lhz [[REG1:[0-9]+]], 4([[REGADDR]])
; ASM32-DAG:   lwz 3, 0([[REGADDR]])
; ASM32-DAG:   slwi 4, [[REG1]], 16
; ASM32-NEXT:  bl .test_byval_6Byte[PR]
; ASM32-NEXT:  nop

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS6, $x2 :: (load (s64) from got)
; 64BIT-DAG:   renamable $x[[REG1:[0-9]+]] = LWZ8 0, killed renamable $x[[REGADDR]] :: (load (s32))
; 64BIT-DAG:   renamable $x[[REG2:[0-9]+]] = LHZ8 4, renamable $x[[REGADDR]] :: (load (s16))
; 64BIT-DAG:   renamable $x3 = RLWINM8 killed renamable $x[[REG2]], 16, 0, 15
; 64BIT-DAG:   renamable $x3 = RLDIMI killed renamable $x3, killed renamable $x[[REG1]], 32, 0
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_6Byte[PR]>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM64:       stdu 1, -112(1)
; ASM64-NEXT:  ld [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64-DAG:   lwz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM64-DAG:   lhz [[REG2:[0-9]+]], 4([[REGADDR]])
; ASM64-DAG:   rlwinm 3, [[REG2]], 16, 0, 15
; ASM64-DAG:   rldimi 3, [[REG1]], 32, 0
; ASM64-NEXT:  bl .test_byval_6Byte[PR]
; ASM64-NEXT:  nop


%struct.S7 = type { [7 x i8] }

@gS7 = external global %struct.S7, align 1

define void @call_test_byval_7Byte() {
entry:
  %call = call zeroext i8 @test_byval_7Byte(ptr byval(%struct.S7) align 1 @gS7)
  ret void
}

declare zeroext i8 @test_byval_7Byte(ptr byval(%struct.S7) align 1)

; CHECK-LABEL: name: call_test_byval_7Byte{{.*}}

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REGADDR:[0-9]+]] = LWZtoc @gS7, $r2 :: (load (s32) from got)
; 32BIT-DAG:   renamable $r3 = LWZ 0, killed renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r[[REG1:[0-9]+]] = LHZ 4, renamable $r[[REGADDR]] :: (load (s16))
; 32BIT-DAG:   renamable $r[[REG2:[0-9]+]] = LBZ 6, renamable $r[[REGADDR]] :: (load (s8))
; 32BIT-DAG:   renamable $r4 = RLWINM killed renamable $r[[REG2]], 8, 16, 23
; 32BIT-DAG:   renamable $r4 = RLWIMI killed renamable $r4, killed renamable $r[[REG1]], 16, 0, 15
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_7Byte[PR]>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_7Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM32:       stwu 1, -64(1)
; ASM32-NEXT:  lwz [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32-DAG:   lwz 3, 0([[REGADDR]])
; ASM32-DAG:   lhz [[REG1:[0-9]+]], 4([[REGADDR]])
; ASM32-DAG:   lbz [[REG2:[0-9]+]], 6([[REGADDR]])
; ASM32-DAG:   rlwinm 4, [[REG2]], 8, 16, 23
; ASM32-DAG:   rlwimi 4, [[REG1]], 16, 0, 15
; ASM32-NEXT:  bl .test_byval_7Byte[PR]
; ASM32-NEXT:  nop

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS7, $x2 :: (load (s64) from got)
; 64BIT-DAG:   renamable $x[[REG1:[0-9]+]] = LWZ8 0, killed renamable $x[[REGADDR]] :: (load (s32))
; 64BIT-DAG:   renamable $x[[REG2:[0-9]+]] = LHZ8 4, renamable $x[[REGADDR]] :: (load (s16))
; 64BIT-DAG:   renamable $x[[REG3:[0-9]+]] = LBZ8 6, renamable $x[[REGADDR]] :: (load (s8))
; 64BIT-DAG:   renamable $x3 = RLWINM8 killed renamable $x[[REG3]], 8, 16, 23
; 64BIT-DAG:   renamable $x3 = RLWIMI8 killed renamable $x3, killed renamable $x[[REG2]], 16, 0, 15
; 64BIT-DAG:   renamable $x3 = RLDIMI killed renamable $x3, killed renamable $x[[REG1]], 32, 0
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_7Byte[PR]>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM64:       stdu 1, -112(1)
; ASM64-NEXT:  ld [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64-DAG:   lwz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM64-DAG:   lhz [[REG2:[0-9]+]], 4([[REGADDR]])
; ASM64-DAG:   lbz [[REG3:[0-9]+]], 6([[REGADDR]])
; ASM64-DAG:   rlwinm 3, [[REG3]], 8, 16, 23
; ASM64-DAG:   rlwimi 3, [[REG2]], 16, 0, 15
; ASM64-DAG:   rldimi 3, [[REG1]], 32, 0
; ASM64-NEXT:  bl .test_byval_7Byte[PR]
; ASM64-NEXT:  nop


%struct.S8 = type { [8 x i8] }

@gS8 = external global %struct.S8, align 1

define void @call_test_byval_8Byte() {
entry:
  %call = call zeroext i8 @test_byval_8Byte(ptr byval(%struct.S8) align 1 @gS8)
  ret void
}

declare zeroext i8 @test_byval_8Byte(ptr byval(%struct.S8) align 1)

; CHECK-LABEL: name: call_test_byval_8Byte{{.*}}

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REGADDR:[0-9]+]] = LWZtoc @gS8, $r2 :: (load (s32) from got)
; 32BIT-DAG:   renamable $r3 = LWZ 0, killed renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r4 = LWZ 4, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_8Byte[PR]>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_8Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM32:       stwu 1, -64(1)
; ASM32-NEXT:  lwz [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32-DAG:   lwz 3, 0([[REGADDR]])
; ASM32-DAG:   lwz 4, 4([[REGADDR]])
; ASM32-NEXT:  bl .test_byval_8Byte[PR]
; ASM32-NEXT:  nop

; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS8, $x2 :: (load (s64) from got)
; 64BIT-NEXT:  renamable $x3 = LD 0, killed renamable $x[[REGADDR]] :: (load (s64))
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_8Byte[PR]>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; ASM64:       stdu 1, -112(1)
; ASM64-NEXT:  ld [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64-NEXT:  std 0, 128(1)
; ASM64-NEXT:  ld 3, 0([[REGADDR]])
; ASM64-NEXT:  bl .test_byval_8Byte[PR]
; ASM64-NEXT:  nop


%struct.S32 = type { [32 x i8] }

@gS32 = external global %struct.S32, align 1

define void @call_test_byval_32Byte() {
entry:
  %call = call zeroext i8 @test_byval_32Byte(ptr byval(%struct.S32) align 1 @gS32)
  ret void
}

; CHECK-LABEL: name: call_test_byval_32Byte{{.*}}

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REGADDR:[0-9]+]] = LWZtoc @gS32, $r2 :: (load (s32) from got)
; 32BIT-DAG:   renamable $r3 = LWZ 0, killed renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r4 = LWZ 4, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r5 = LWZ 8, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r6 = LWZ 12, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r7 = LWZ 16, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r8 = LWZ 20, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r9 = LWZ 24, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r10 = LWZ 28, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_32Byte>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit $r5, implicit $r6, implicit $r7, implicit $r8, implicit $r9, implicit $r10, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_32Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM32:       stwu 1, -64(1)
; ASM32-NEXT:  lwz [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32-DAG:   lwz 3, 0([[REGADDR]])
; ASM32-DAG:   lwz 4, 4([[REGADDR]])
; ASM32-DAG:   lwz 5, 8([[REGADDR]])
; ASM32-DAG:   lwz 6, 12([[REGADDR]])
; ASM32-DAG:   lwz 7, 16([[REGADDR]])
; ASM32-DAG:   lwz 8, 20([[REGADDR]])
; ASM32-DAG:   lwz 9, 24([[REGADDR]])
; ASM32-DAG:   lwz 10, 28([[REGADDR]])
; ASM32-NEXT:  bl .test_byval_32Byte
; ASM32-NEXT:  nop

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS32, $x2 :: (load (s64) from got)
; 64BIT-DAG:   renamable $x3 = LD 0, killed renamable $x[[REGADDR]] :: (load (s64))
; 64BIT-DAG:   renamable $x4 = LD 8, renamable $x[[REGADDR]] :: (load (s64))
; 64BIT-DAG:   renamable $x5 = LD 16, renamable $x[[REGADDR]] :: (load (s64))
; 64BIT-DAG:   renamable $x6 = LD 24, renamable $x[[REGADDR]] :: (load (s64))
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_32Byte>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x4, implicit $x5, implicit $x6, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; ASM64:       stdu 1, -112(1)
; ASM64-NEXT:  ld [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64-DAG:   ld 3, 0([[REGADDR]])
; ASM64-DAG:   ld 4, 8([[REGADDR]])
; ASM64-DAG:   ld 5, 16([[REGADDR]])
; ASM64-DAG:   ld 6, 24([[REGADDR]])
; ASM64-NEXT:  bl .test_byval_32Byte
; ASM64-NEXT:  nop

define zeroext i8 @test_byval_32Byte(ptr byval(%struct.S32) align 1 %s) {
entry:
  %arrayidx = getelementptr inbounds %struct.S32, ptr %s, i32 0, i32 0, i32 21
  %0 = load i8, ptr %arrayidx, align 1
  ret i8 %0
}

; The ByVal handling produces dead stores. See `LowerFormalArguments_AIX` for
; details on why.

; CHECK-LABEL: name:            test_byval_32Byte

; 32BIT:      fixedStack:
; 32BIT-NEXT:   - { id: 0, type: default, offset: 24, size: 32, alignment: 8, stack-id: default,
; 32BIT-NEXT:       isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,

; 32BIT:      bb.0.entry:
; 32BIT-NEXT:   liveins: $r3, $r4, $r5, $r6, $r7, $r8, $r9, $r10
; 32BIT:        STW killed renamable $r8,  20, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 20
; 32BIT-DAG:    STW killed renamable $r3,   0, %fixed-stack.0 :: (store (s32) into %fixed-stack.0
; 32BIT-DAG:    STW killed renamable $r4,   4, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 4
; 32BIT-DAG:    STW killed renamable $r5,   8, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 8
; 32BIT-DAG:    STW killed renamable $r6,  12, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 12
; 32BIT-DAG:    STW killed renamable $r7,  16, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 16
; 32BIT:        renamable $r3 = LBZ 21, %fixed-stack.0 :: (dereferenceable load (s8)
; 32BIT-DAG:    STW killed renamable $r9,  24, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 24
; 32BIT-DAG:    STW killed renamable $r10, 28, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 28
; 32BIT:        BLR

; 64BIT:      fixedStack:
; 64BIT-NEXT:   - { id: 0, type: default, offset: 48, size: 32, alignment: 16, stack-id: default,
; 64BIT-NEXT:       isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,

; 64BIT:      bb.0.entry:
; 64BIT-NEXT:   liveins: $x3, $x4, $x5, $x6
; 64BIT:        STD killed renamable $x5, 16, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 16
; 64BIT-DAG:    STD killed renamable $x3, 0, %fixed-stack.0 :: (store (s64) into %fixed-stack.0
; 64BIT-NEXT:   renamable $x3 = LBZ8 21, %fixed-stack.0 :: (dereferenceable load (s8)
; 64BIT-DAG:    STD killed renamable $x4, 8, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 8
; 64BIT-DAG:    STD killed renamable $x6, 24, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 24
; 64BIT-NEXT:   BLR8

; CHECKASM-LABEL: .test_byval_32Byte:

; ASM32:       stw 8, 44(1)
; ASM32:       stw 3, 24(1)
; ASM32-DAG:   lbz 3, 45(1)
; ASM32-DAG:   stw 4, 28(1)
; ASM32-DAG:   stw 5, 32(1)
; ASM32-DAG:   stw 6, 36(1)
; ASM32-DAG:   stw 7, 40(1)
; ASM32-DAG:   stw 9, 48(1)
; ASM32-DAG:   stw 10, 52(1)
; ASM32-NEXT:  blr

; ASM64:       std 5, 64(1)
; ASM64:       std 3, 48(1)
; ASM64-DAG:   lbz 3, 69(1)
; ASM64-DAG:   std 4, 56(1)
; ASM64-DAG:   std 6, 72(1)
; ASM64-NEXT:  blr

%struct.S31 = type <{ float, i32, i64, double, i32, i16, i8 }>

@gS31 = external global %struct.S31, align 1

define void @call_test_byval_31Byte() {
entry:
  %call = call double @test_byval_31Byte(ptr byval(%struct.S31) align 1 @gS31)
  ret void
}


; CHECK-LABEL: name: call_test_byval_31Byte{{.*}}

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:  renamable $r[[REGADDR:[0-9]+]] = LWZtoc @gS31, $r2 :: (load (s32) from got)
; 32BIT-DAG:   renamable $r3 = LWZ 0, killed renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r4 = LWZ 4, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r5 = LWZ 8, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r6 = LWZ 12, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r7 = LWZ 16, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r8 = LWZ 20, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r9 = LWZ 24, renamable $r[[REGADDR]] :: (load (s32))
; 32BIT-DAG:   renamable $r[[REG:[0-9]+]] = LHZ 28, renamable $r[[REGADDR]] :: (load (s16))
; 32BIT-DAG:   renamable $r10 = LBZ 30, renamable $r[[REGADDR]] :: (load (s8))
; 32BIT-DAG:   renamable $r10 = RLWINM killed renamable $r10, 8, 16, 23
; 32BIT-DAG:   renamable $r10 = RLWIMI killed renamable $r10, killed renamable $r[[REG]], 16, 0, 15
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_31Byte>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit $r5, implicit $r6, implicit $r7, implicit $r8, implicit $r9, implicit $r10, implicit $r2, implicit-def $r1
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_31Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM32:       stwu 1, -64(1)
; ASM32-NEXT:  lwz [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32-DAG:   lwz 3, 0([[REGADDR]])
; ASM32-DAG:   lwz 4, 4([[REGADDR]])
; ASM32-DAG:   lwz 5, 8([[REGADDR]])
; ASM32-DAG:   lwz 6, 12([[REGADDR]])
; ASM32-DAG:   lwz 7, 16([[REGADDR]])
; ASM32-DAG:   lwz 8, 20([[REGADDR]])
; ASM32-DAG:   lwz 9, 24([[REGADDR]])
; ASM32-DAG:   lbz 10, 30([[REGADDR]])
; ASM32-DAG:   lhz [[REG:[0-9]+]], 28([[REGADDR]])
; ASM32-DAG:   rlwinm 10, 10, 8, 16, 23
; ASM32-DAG:   rlwimi 10, [[REG]], 16, 0, 15
; ASM32-NEXT:  bl .test_byval_31Byte
; ASM32-NEXT:  nop

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS31, $x2 :: (load (s64) from got)
; 64BIT-DAG:   renamable $x3 = LD 0, killed renamable $x[[REGADDR]] :: (load (s64))
; 64BIT-DAG:   renamable $x4 = LD 8, renamable $x[[REGADDR]] :: (load (s64))
; 64BIT-DAG:   renamable $x5 = LD 16, renamable $x[[REGADDR]] :: (load (s64))
; 64BIT-DAG:   renamable $x[[REG1:[0-9]+]] = LWZ8 24, renamable $x[[REGADDR]] :: (load (s32))
; 64BIT-DAG:   renamable $x[[REG2:[0-9]+]] = LHZ8 28, renamable $x[[REGADDR]] :: (load (s16))
; 64BIT-DAG:   renamable $x[[REG3:[0-9]+]] = LBZ8 30, renamable $x[[REGADDR]] :: (load (s8))
; 64BIT-DAG:   renamable $x6 = RLWINM8 killed renamable $x[[REG3]], 8, 16, 23
; 64BIT-DAG:   renamable $x6 = RLWIMI8 killed renamable $x6, killed renamable $x[[REG2]], 16, 0, 15
; 64BIT-DAG:   renamable $x6 = RLDIMI killed renamable $x6, killed renamable $x[[REG1]], 32, 0
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_31Byte>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x4, implicit $x5, implicit $x6, implicit $x2, implicit-def $r1
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; ASM64:       stdu 1, -112(1)
; ASM64-NEXT:  ld [[REGADDR:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64-DAG:   ld 3, 0([[REGADDR]])
; ASM64-DAG:   ld 4, 8([[REGADDR]])
; ASM64-DAG:   ld 5, 16([[REGADDR]])
; ASM64-DAG:   lwz [[REG1:[0-9]+]], 24([[REGADDR]])
; ASM64-DAG:   lhz [[REG2:[0-9]+]], 28([[REGADDR]])
; ASM64-DAG:   lbz [[REG3:[0-9]+]], 30([[REGADDR]])
; ASM64-DAG:   rlwinm 6, [[REG3]], 8, 16, 23
; ASM64-DAG:   rlwimi 6, [[REG2]], 16, 0, 15
; ASM64-DAG:   rldimi 6, [[REG1]], 32, 0
; ASM64-NEXT:  bl .test_byval_31Byte
; ASM64-NEXT:  nop



define double @test_byval_31Byte(ptr byval(%struct.S31) align 1 %s) {
entry:
  %gep = getelementptr inbounds %struct.S31, ptr %s, i32 0, i32 3
  %load = load double, ptr %gep, align 1
  ret double %load
}

; CHECK-LABEL: name:            test_byval_31Byte

; 32BIT:      fixedStack:
; 32BIT-NEXT:   - { id: 0, type: default, offset: 24, size: 32, alignment: 8, stack-id: default,
; 32BIT-NEXT:       isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,

; 32BIT:      bb.0.entry:
; 32BIT-NEXT:   liveins: $r3, $r4, $r5, $r6, $r7, $r8, $r9, $r10
; 32BIT-DAG:    STW killed renamable $r3,   0, %fixed-stack.0 :: (store (s32) into %fixed-stack.0
; 32BIT-DAG:    STW killed renamable $r4,   4, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 4
; 32BIT-DAG:    STW killed renamable $r5,   8, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 8
; 32BIT-DAG:    STW killed renamable $r6,  12, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 12
; 32BIT-DAG:    STW killed renamable $r7,  16, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 16
; 32BIT-DAG:    STW killed renamable $r8,  20, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 20
; 32BIT-NEXT:   renamable $f1 = LFD 16, %fixed-stack.0 :: (dereferenceable load (s64)
; 32BIT-DAG:    STW killed renamable $r9,  24, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 24
; 32BIT-DAG:    STW killed renamable $r10, 28, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 28
; 32BIT-NEXT:   BLR

; 64BIT:      fixedStack:
; 64BIT-NEXT:   - { id: 0, type: default, offset: 48, size: 32, alignment: 16, stack-id: default,
; 64BIT-NEXT:       isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,

; 64BIT:      bb.0.entry:
; 64BIT-NEXT:   liveins: $x3, $x4, $x5, $x6
; 64BIT-DAG:    STD killed renamable $x3,  0, %fixed-stack.0 :: (store (s64) into %fixed-stack.0
; 64BIT-DAG:    STD killed renamable $x5, 16, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 16
; 64BIT-NEXT:   renamable $f1 = LFD 16, %fixed-stack.0 :: (dereferenceable load (s64)
; 64BIT-DAG:    STD killed renamable $x4,  8, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 8
; 64BIT-DAG:    STD killed renamable $x6, 24, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 24
; 64BIT-NEXT:   BLR8

; ASM32-LABEL: .test_byval_31Byte:

; ASM32-DAG:      stw 8, 44(1)
; ASM32:          stw 7, 40(1)
; ASM32-DAG:      lfd 1, 40(1)
; ASM32-DAG:      stw 3, 24(1)
; ASM32-DAG:      stw 4, 28(1)
; ASM32-DAG:      stw 5, 32(1)
; ASM32-DAG:      stw 6, 36(1)
; ASM32-DAG:      stw 9, 48(1)
; ASM32-DAG:      stw 10, 52(1)
; ASM32-NEXT:     blr

; ASM64:          std 5, 64(1)
; ASM64:          lfd 1, 64(1)
; ASM64-DAG:      std 3, 48(1)
; ASM64-DAG:      std 4, 56(1)
; ASM64-DAG:      std 6, 72(1)
; ASM64-NEXT:     blr

%struct.F = type { float, float, float }

define i32 @call_test_byval_homogeneous_float_struct() {
entry:
  %s = alloca %struct.F, align 8
  call void @llvm.memset.p0.i32(ptr align 4 %s, i8 0, i32 12, i1 false)
  %call = call i32 @test_byval_homogeneous_float_struct(ptr byval(%struct.F) align 4 %s)
  ret i32 %call
}

declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg)

declare i32 @test_byval_homogeneous_float_struct(ptr byval(%struct.F) align 4)

; CHECK-LABEL: name: call_test_byval_homogeneous_float_struct{{.*}}

; 32BIT:       ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-DAG:   renamable $r4 = LWZ 4, %stack.0.s :: (load (s32) from %stack.0.s + 4)
; 32BIT-DAG:   renamable $r5 = LWZ 8, %stack.0.s :: (load (s32) from %stack.0.s + 8, align 8)
; 32BIT-DAG:   $r3 = LI 0
; 32BIT-NEXT:  BL_NOP <mcsymbol .test_byval_homogeneous_float_struct[PR]>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit $r5, implicit $r2, implicit-def $r1, implicit-def $r3
; 32BIT-NEXT:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_homogeneous_float_struct:

; ASM32:       stwu 1, -80(1)
; ASM32-DAG:   lwz 4, 68(1)
; ASM32-DAG:   lwz 5, 72(1)
; ASM32-DAG:   stw 3, 64(1)
; ASM32-NEXT:  bl .test_byval_homogeneous_float_struct[PR]
; ASM32-NEXT:  nop

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; 64BIT:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT:       renamable $x3 = LWZ8 8, %stack.0.s :: (load (s32) from %stack.0.s + 8, align 8)
; 64BIT-NEXT:  renamable $x4 = RLDICR killed renamable $x3, 32, 31
; 64BIT-NEXT:  $x3 = LI8 0
; 64BIT-NEXT:  BL8_NOP <mcsymbol .test_byval_homogeneous_float_struct[PR]>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x4, implicit $x2, implicit-def $r1, implicit-def $x3
; 64BIT-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM64:       stdu 1, -128(1)
; ASM64:       lwz 3, 120(1)
; ASM64-NEXT:  sldi 4, 3, 32
; ASM64-NEXT:  li 3, 0
; ASM64-NEXT:  bl .test_byval_homogeneous_float_struct[PR]
; ASM64-NEXT:  nop

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp -mcpu=pwr4 \
; RUN: -mattr=-altivec -verify-machineinstrs < %s | \
; RUN: FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefix=ASM %s

%struct.S5 = type { [5 x i8] }

@gS5 = external global %struct.S5, align 1

define void @call_test_byval_5Byte() {
entry:
  %call = call zeroext i8 @test_byval_5Byte(%struct.S5* byval(%struct.S5) align 1 @gS5)
  ret void
}

; CHECK-LABEL: name: call_test_byval_5Byte{{.*}}

; ASM-LABEL: .call_test_byval_5Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; CHECK:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; CHECK-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS5, $x2 :: (load 8 from got)
; CHECK-DAG:   renamable $x[[REG1:[0-9]+]] = LWZ8 0, killed renamable $x[[REGADDR]] :: (load 4)
; CHECK-DAG:   renamable $x[[REG2:[0-9]+]] = LBZ8 4, renamable $x[[REGADDR]] :: (load 1)
; CHECK-DAG:   renamable $x3 = RLWINM8 killed renamable $x[[REG2]], 24, 0, 7
; CHECK-DAG:   renamable $x3 = RLDIMI killed renamable $x3, killed renamable $x[[REG1]], 32, 0
; CHECK-NEXT:  BL8_NOP <mcsymbol .test_byval_5Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; CHECK-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM:       stdu 1, -112(1)
; ASM-NEXT:  ld [[REGADDR:[0-9]+]], LC{{[0-9]+}}(2)
; ASM-DAG:   lwz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM-DAG:   lbz [[REG2:[0-9]+]], 4([[REGADDR]])
; ASM-DAG:   rlwinm 3, [[REG2]], 24, 0, 7
; ASM-DAG:   rldimi 3, [[REG1]], 32, 0
; ASM-NEXT:  bl .test_byval_5Byte
; ASM-NEXT:  nop


define zeroext i8 @test_byval_5Byte(%struct.S5* byval(%struct.S5) align 1 %s) {
entry:
  %arrayidx = getelementptr inbounds %struct.S5, %struct.S5* %s, i32 0, i32 0, i32 4
  %0 = load i8, i8* %arrayidx, align 1
  ret i8 %0
}

; CHECK-LABEL: name:            test_byval_5Byte

; CHECK:      fixedStack:
; CHECK-NEXT:   - { id: 0, type: default, offset: 48, size: 8, alignment: 16,
; CHECK:        bb.0.entry:
; CHECK-NEXT:     liveins: $x3
; CHECK:          STD killed renamable $x3, 0, %fixed-stack.0 :: (store 8 into %fixed-stack.0, align 16)
; CHECK-NEXT:     renamable $x3 = LBZ8 4, %fixed-stack.0 :: (dereferenceable load 1

; CHECKASM-LABEL: .test_byval_5Byte:

; ASM:       std 3, 48(1)
; ASM-NEXT:  lbz 3, 52(1)
; ASM-NEXT:  blr


%struct.S6 = type { [6 x i8] }

@gS6 = external global %struct.S6, align 1

define void @call_test_byval_6Byte() {
entry:
  %call = call zeroext i8 @test_byval_6Byte(%struct.S6* byval(%struct.S6) align 1 @gS6)
  ret void
}

; CHECK-LABEL: name: call_test_byval_6Byte{{.*}}

; ASM-LABEL: .call_test_byval_6Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; CHECK:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; CHECK-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS6, $x2 :: (load 8 from got)
; CHECK-DAG:   renamable $x[[REG1:[0-9]+]] = LWZ8 0, killed renamable $x[[REGADDR]] :: (load 4)
; CHECK-DAG:   renamable $x[[REG2:[0-9]+]] = LHZ8 4, renamable $x[[REGADDR]] :: (load 2)
; CHECK-DAG:   renamable $x3 = RLWINM8 killed renamable $x[[REG2]], 16, 0, 15
; CHECK-DAG:   renamable $x3 = RLDIMI killed renamable $x3, killed renamable $x[[REG1]], 32, 0
; CHECK-NEXT:  BL8_NOP <mcsymbol .test_byval_6Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; CHECK-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM:       stdu 1, -112(1)
; ASM-NEXT:  ld [[REGADDR:[0-9]+]], LC{{[0-9]+}}(2)
; ASM-DAG:   lwz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM-DAG:   lhz [[REG2:[0-9]+]], 4([[REGADDR]])
; ASM-DAG:   rlwinm 3, [[REG2]], 16, 0, 15
; ASM-DAG:   rldimi 3, [[REG1]], 32, 0
; ASM-NEXT:  bl .test_byval_6Byte
; ASM-NEXT:  nop


define zeroext i8 @test_byval_6Byte(%struct.S6* byval(%struct.S6) align 1 %s) {
entry:
  %arrayidx = getelementptr inbounds %struct.S6, %struct.S6* %s, i32 0, i32 0, i32 5
  %0 = load i8, i8* %arrayidx, align 1
  ret i8 %0
}

; CHECK-LABEL: name:            test_byval_6Byte

; CHECK:      fixedStack:
; CHECK-NEXT:   - { id: 0, type: default, offset: 48, size: 8, alignment: 16,
; CHECK:        bb.0.entry:
; CHECK-NEXT:     liveins: $x3
; CHECK:          STD killed renamable $x3, 0, %fixed-stack.0 :: (store 8 into %fixed-stack.0, align 16)
; CHECK-NEXT:     renamable $x3 = LBZ8 5, %fixed-stack.0 :: (dereferenceable load 1

; CHECKASM-LABEL: .test_byval_6Byte:

; ASM:       std 3, 48(1)
; ASM-NEXT:  lbz 3, 53(1)
; ASM-NEXT:  blr

%struct.S7 = type { [7 x i8] }

@gS7 = external global %struct.S7, align 1

define void @call_test_byval_7Byte() {
entry:
  %call = call zeroext i8 @test_byval_7Byte(%struct.S7* byval(%struct.S7) align 1 @gS7)
  ret void
}

; CHECK-LABEL: name: call_test_byval_7Byte{{.*}}

; ASM-LABEL: .call_test_byval_7Byte:

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; CHECK:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; CHECK-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS7, $x2 :: (load 8 from got)
; CHECK-DAG:   renamable $x[[REG1:[0-9]+]] = LWZ8 0, killed renamable $x[[REGADDR]] :: (load 4)
; CHECK-DAG:   renamable $x[[REG2:[0-9]+]] = LHZ8 4, renamable $x[[REGADDR]] :: (load 2)
; CHECK-DAG:   renamable $x[[REG3:[0-9]+]] = LBZ8 6, renamable $x[[REGADDR]] :: (load 1)
; CHECK-DAG:   renamable $x3 = RLWINM8 killed renamable $x[[REG3]], 8, 16, 23
; CHECK-DAG:   renamable $x3 = RLWIMI8 killed renamable $x3, killed renamable $x[[REG2]], 16, 0, 15
; CHECK-DAG:   renamable $x3 = RLDIMI killed renamable $x3, killed renamable $x[[REG1]], 32, 0
; CHECK-NEXT:  BL8_NOP <mcsymbol .test_byval_7Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; CHECK-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; ASM:       stdu 1, -112(1)
; ASM-NEXT:  ld [[REGADDR:[0-9]+]], LC{{[0-9]+}}(2)
; ASM-DAG:   lwz [[REG1:[0-9]+]], 0([[REGADDR]])
; ASM-DAG:   lhz [[REG2:[0-9]+]], 4([[REGADDR]])
; ASM-DAG:   lbz [[REG3:[0-9]+]], 6([[REGADDR]])
; ASM-DAG:   rlwinm 3, [[REG3]], 8, 16, 23
; ASM-DAG:   rlwimi 3, [[REG2]], 16, 0, 15
; ASM-DAG:   rldimi 3, [[REG1]], 32, 0
; ASM-NEXT:  bl .test_byval_7Byte
; ASM-NEXT:  nop


define zeroext i8 @test_byval_7Byte(%struct.S7* byval(%struct.S7) align 1 %s) {
entry:
  %arrayidx = getelementptr inbounds %struct.S7, %struct.S7* %s, i32 0, i32 0, i32 6
  %0 = load i8, i8* %arrayidx, align 1
  ret i8 %0
}

; CHECK-LABEL: name:            test_byval_7Byte

; CHECK:      fixedStack:
; CHECK-NEXT:   - { id: 0, type: default, offset: 48, size: 8, alignment: 16,
; CHECK:        bb.0.entry:
; CHECK-NEXT:     liveins: $x3
; CHECK:          STD killed renamable $x3, 0, %fixed-stack.0 :: (store 8 into %fixed-stack.0, align 16)
; CHECK-NEXT:     renamable $x3 = LBZ8 6, %fixed-stack.0 :: (dereferenceable load 1

; CHECKASM-LABEL: .test_byval_7Byte:

; ASM:       std 3, 48(1)
; ASM-NEXT:  lbz 3, 54(1)
; ASM-NEXT:  blr


%struct.S8 = type { [8 x i8] }

@gS8 = external global %struct.S8, align 1

define void @call_test_byval_8Byte() {
entry:
  %call = call zeroext i8 @test_byval_8Byte(%struct.S8* byval(%struct.S8) align 1 @gS8)
  ret void
}

; CHECK-LABEL: name: call_test_byval_8Byte{{.*}}

; ASM-LABEL: .call_test_byval_8Byte:

; CHECK:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; CHECK-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS8, $x2 :: (load 8 from got)
; CHECK-NEXT:  renamable $x3 = LD 0, killed renamable $x[[REGADDR]] :: (load 8)
; CHECK-NEXT:  BL8_NOP <mcsymbol .test_byval_8Byte>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; CHECK-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; ASM:       stdu 1, -112(1)
; ASM-NEXT:  ld [[REGADDR:[0-9]+]], LC{{[0-9]+}}(2)
; ASM-NEXT:  ld 3, 0([[REGADDR]])
; ASM-NEXT:  bl .test_byval_8Byte
; ASM-NEXT:  nop


define zeroext i8 @test_byval_8Byte(%struct.S8* byval(%struct.S8) align 1 %s) {
entry:
  %arrayidx = getelementptr inbounds %struct.S8, %struct.S8* %s, i32 0, i32 0, i32 7
  %0 = load i8, i8* %arrayidx, align 1
  ret i8 %0
}

; CHECK-LABEL: name:            test_byval_8Byte

; CHECK:      fixedStack:
; CHECK-NEXT:   - { id: 0, type: default, offset: 48, size: 8, alignment: 16,
; CHECK:        bb.0.entry:
; CHECK-NEXT:     liveins: $x3
; CHECK:          renamable $x[[SCRATCH:[0-9]+]] = COPY $x3
; CHECK-DAG:      renamable $x3 = RLDICL $x3, 0, 56
; CHECK-DAG:      STD killed renamable $x[[SCRATCH]], 0, %fixed-stack.0 :: (store 8 into %fixed-stack.0, align 16)


; CHECKASM-LABEL: .test_byval_8Byte:

; ASM:       mr [[SCRATCH:[0-9]+]], 3
; ASM-DAG:   clrldi  3, 3, 56
; ASM-DAG:   std [[SCRATCH]], 48(1)
; ASM-NEXT:  blr

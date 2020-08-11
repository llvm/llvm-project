; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp -mcpu=pwr4 \
; RUN: -mattr=-altivec -verify-machineinstrs < %s | \
; RUN: FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefix=ASM %s

%struct.S5 = type { [5 x i8] }

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


%struct.S64 = type { [64 x i8] }

@gS64 = external global %struct.S64, align 1

define void @call_test_byval_64Byte() {
entry:
  call void @test_byval_64Byte(%struct.S64* byval(%struct.S64) align 1 @gS64)
  ret void
}

declare void @test_byval_64Byte(%struct.S64* byval(%struct.S64) align 1)

; CHECK-LABEL: name: call_test_byval_64Byte{{.*}}

; The DAG block permits some invalid inputs for the benefit of allowing more valid orderings.
; CHECK:       ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; CHECK-NEXT:  renamable $x[[REGADDR:[0-9]+]] = LDtoc @gS64, $x2 :: (load 8 from got)
; CHECK-DAG:   renamable $x3 = LD 0, killed renamable $x[[REGADDR]] :: (load 8)
; CHECK-DAG:   renamable $x4 = LD 8, renamable $x[[REGADDR]] :: (load 8)
; CHECK-DAG:   renamable $x5 = LD 16, renamable $x[[REGADDR]] :: (load 8)
; CHECK-DAG:   renamable $x6 = LD 24, renamable $x[[REGADDR]] :: (load 8)
; CHECK-DAG:   renamable $x7 = LD 32, renamable $x[[REGADDR]] :: (load 8)
; CHECK-DAG:   renamable $x8 = LD 40, renamable $x[[REGADDR]] :: (load 8)
; CHECK-DAG:   renamable $x9 = LD 48, renamable $x[[REGADDR]] :: (load 8)
; CHECK-DAG:   renamable $x10 = LD 56, renamable $x[[REGADDR]] :: (load 8)
; CHECK-NEXT:  BL8_NOP <mcsymbol .test_byval_64Byte[PR]>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x4, implicit $x5, implicit $x6, implicit $x7, implicit $x8, implicit $x9, implicit $x10, implicit $x2, implicit-def $r1
; CHECK-NEXT:  ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .test_byval_64Byte:

; ASM:         stdu 1, -112(1)
; ASM-NEXT:    ld [[REG:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM-DAG:     ld 3, 0([[REG]])
; ASM-DAG:     ld 4, 8([[REG]])
; ASM-DAG:     ld 5, 16([[REG]])
; ASM-DAG:     ld 6, 24([[REG]])
; ASM-DAG:     ld 7, 32([[REG]])
; ASM-DAG:     ld 8, 40([[REG]])
; ASM-DAG:     ld 9, 48([[REG]])
; ASM-DAG:     ld 10, 56([[REG]])
; ASM-NEXT:    bl .test_byval_64Byte
; ASM-NEXT:    nop

; RUN: llc -verify-machineinstrs -stop-before=ppc-vsx-copy \
; RUN:  -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=CHECK,32BIT %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=CHECKASM,ASM32BIT %s

; RUN: llc -verify-machineinstrs -stop-before=ppc-vsx-copy \
; RUN:  -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=CHECK,64BIT %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=CHECKASM,ASM64BIT %s

%struct_S1 = type { i8 }

@gS1 = external global %struct_S1, align 1

define void @call_test_byval_mem1() {
entry:
  %call = call zeroext i8 @test_byval_mem1(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, %struct_S1* byval(%struct_S1) align 1 @gS1)
  ret void
}


; CHECKASM-LABEL: .call_test_byval_mem1:

; ASM32BIT:       stwu 1, -64(1)
; ASM32BIT:       lwz [[REG1:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32BIT:       lbz [[REG2:[0-9]+]], 0([[REG1]])
; ASM32BIT:       stb [[REG2]], 56(1)
; ASM32BIT:       bl .test_byval_mem1
; ASM32BIT:       addi 1, 1, 64

; ASM64BIT:       stdu 1, -128(1)
; ASM64BIT:       ld [[REG1:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64BIT:       lbz [[REG2:[0-9]+]], 0([[REG1]])
; ASM64BIT:       stb [[REG2]], 112(1)
; ASM64BIT:       bl .test_byval_mem1
; ASM64BIT:       addi 1, 1, 128

define zeroext  i8 @test_byval_mem1(i32, i32, i32, i32, i32, i32, i32, i32, %struct_S1* byval(%struct_S1) align 1 %s) {
entry:
  %gep = getelementptr inbounds %struct_S1, %struct_S1* %s, i32 0, i32 0
  %load = load i8, i8* %gep, align 1
  ret i8 %load
}

; CHECK-LABEL: name:            test_byval_mem1

; 32BIT:       fixedStack:
; 32BIT-NEXT:    - { id: 0, type: default, offset: 56, size: 4, alignment: 8, stack-id: default,
; 32BIT:       bb.0.entry:
; 32BIT-NEXT:    %[[VAL:[0-9]+]]:gprc = LBZ 0, %fixed-stack.0
; 32BIT-NEXT:    $r3 = COPY %[[VAL]]
; 32BIT-NEXT:    BLR

; 64BIT:       fixedStack:
; 64BIT-NEXT:    - { id: 0, type: default, offset: 112, size: 8, alignment: 16, stack-id: default,
; 64BIT:       bb.0.entry:
; 64BIT-NEXT:    %[[VAL:[0-9]+]]:g8rc = LBZ8 0, %fixed-stack.0
; 64BIT-NEXT:    $x3 = COPY %[[VAL]]
; 64BIT-NEXT:    BLR8


%struct_S256 = type { [256 x i8] }

@gS256 = external global %struct_S256, align 1

define void @call_test_byval_mem2() {
entry:
  %call = call zeroext i8 @test_byval_mem2(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, %struct_S256* byval(%struct_S256) align 1 @gS256)
  ret void
}


; CHECK-LABEL:    name: call_test_byval_mem2

; Confirm the expected memcpy call is independent of the call to test_byval_mem2.
; 32BIT:          ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:     %0:gprc = nuw ADDI $r1, 56
; 32BIT-NEXT:     %1:gprc = LWZtoc @gS256, $r2 :: (load 4 from got)
; 32BIT-NEXT:     %2:gprc = LI 256
; 32BIT-DAG:      $r3 = COPY %0
; 32BIT-DAG:      $r4 = COPY %1
; 32BIT-DAG:      $r5 = COPY %2
; 32BIT-NEXT:     BL_NOP &".memcpy[PR]", csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit $r5, implicit $r2, implicit-def $r1, implicit-def $r3
; 32BIT-NEXT:     ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT:          ADJCALLSTACKDOWN 312, 0, implicit-def dead $r1, implicit $r1
; 32BIT-DAG:      $r3 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r4 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r5 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r6 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r7 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r8 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r9 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r10 = COPY %{{[0-9]+}}
; 32BIT-NEXT:     BL_NOP <mcsymbol .test_byval_mem2>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit $r5, implicit $r6, implicit $r7, implicit $r8, implicit $r9, implicit $r10, implicit $r2, implicit-def $r1
; 32BIT-NEXT:     ADJCALLSTACKUP 312, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_mem2:

; ASM32BIT:       stwu 1, -320(1)
; ASM32BIT-DAG:   addi 3, 1, 56
; ASM32BIT-DAG:   lwz 4, L..C{{[0-9]+}}(2)
; ASM32BIT-DAG:   li 5, 256
; ASM32BIT-NEXT:  bl .memcpy[PR]
; ASM32BIT:       bl .test_byval_mem2
; ASM32BIT:       addi 1, 1, 320

; Confirm the expected memcpy call is independent of the call to test_byval_mem2.
; 64BIT:          ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:     %0:g8rc = nuw ADDI8 $x1, 112
; 64BIT-NEXT:     %1:g8rc = LDtoc @gS256, $x2 :: (load 8 from got)
; 64BIT-NEXT:     %2:g8rc = LI8 256
; 64BIT-DAG:      $x3 = COPY %0
; 64BIT-DAG:      $x4 = COPY %1
; 64BIT-DAG:      $x5 = COPY %2
; 64BIT-NEXT:     BL8_NOP &".memcpy[PR]", csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x4, implicit $x5, implicit $x2, implicit-def $r1, implicit-def $x3
; 64BIT-NEXT:     ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT:          ADJCALLSTACKDOWN 368, 0, implicit-def dead $r1, implicit $r1
; 64BIT-DAG:      $x3 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x4 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x5 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x6 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x7 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x8 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x9 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x10 = COPY %{{[0-9]+}}
; 64BIT-NEXT:     BL8_NOP <mcsymbol .test_byval_mem2>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x4, implicit $x5, implicit $x6, implicit $x7, implicit $x8, implicit $x9, implicit $x10, implicit $x2, implicit-def $r1
; 64BIT-NEXT:     ADJCALLSTACKUP 368, 0, implicit-def dead $r1, implicit $r1

; ASM64BIT:       stdu 1, -368(1)
; ASM64BIT-DAG:   addi 3, 1, 112
; ASM64BIT-DAG:   ld 4, L..C{{[0-9]+}}(2)
; ASM64BIT-DAG:   li 5, 256
; ASM64BIT-NEXT:  bl .memcpy[PR]
; ASM64BIT:       bl .test_byval_mem2
; ASM64BIT:       addi 1, 1, 368


define zeroext i8 @test_byval_mem2(i32, i32, i32, i32, i32, i32, i32, i32, %struct_S256* byval(%struct_S256) align 1 %s) {
entry:
  %gep = getelementptr inbounds %struct_S256, %struct_S256* %s, i32 0, i32 0, i32 255
  %load = load i8, i8* %gep, align 1
  ret i8 %load
}

; CHECK-LABEL: name:            test_byval_mem2

; 32BIT:      fixedStack:
; 32BIT-NEXT:   - { id: 0, type: default, offset: 56, size: 256, alignment: 8, stack-id: default,
; 32BIT:      bb.0.entry:
; 32BIT-NEXT:   %[[VAL:[0-9]+]]:gprc = LBZ 255, %fixed-stack.0
; 32BIT-NEXT:   $r3 = COPY %[[VAL]]
; 32BIT-NEXT:   BLR

; 64BIT:      fixedStack:
; 64BIT-NEXT:   - { id: 0, type: default, offset: 112, size: 256, alignment: 16, stack-id: default,
; 64BIT:      bb.0.entry:
; 64BIT-NEXT:   %[[VAL:[0-9]+]]:g8rc = LBZ8 255, %fixed-stack.0
; 64BIT-NEXT:   $x3 = COPY %[[VAL]]
; 64BIT-NEXT:   BLR8

%struct_S57 = type { [57 x i8] }

@gS57 = external global %struct_S57, align 1

define void @call_test_byval_mem3() {
entry:
  call void @test_byval_mem3(i32 42, float 0x40091EB860000000, %struct_S57* byval(%struct_S57) align 1 @gS57)
  ret void
}

; CHECK-LABEL:    name: call_test_byval_mem3

; Confirm the expected memcpy call is independent of the call to test_byval_mem3.
; 32BIT:          ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:     %0:gprc_and_gprc_nor0 = LWZtoc @gS57, $r2 :: (load 4 from got)
; 32BIT-NEXT:     %1:gprc = nuw ADDI %0, 24
; 32BIT-NEXT:     %2:gprc = nuw ADDI $r1, 56
; 32BIT-NEXT:     %3:gprc = LI 33
; 32BIT-DAG:      $r3 = COPY %2
; 32BIT-DAG:      $r4 = COPY %1
; 32BIT-DAG:      $r5 = COPY %3
; 32BIT-NEXT:     BL_NOP &".memcpy[PR]", csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit $r5, implicit $r2, implicit-def $r1, implicit-def $r3
; 32BIT-NEXT:     ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT:          ADJCALLSTACKDOWN 92, 0, implicit-def dead $r1, implicit $r1
; 32BIT-DAG:      $r3 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $f1 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r5 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r6 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r7 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r8 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r9 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r10 = COPY %{{[0-9]+}}
; 32BIT-NEXT:     BL_NOP <mcsymbol .test_byval_mem3>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $f1, implicit $r5, implicit $r6, implicit $r7, implicit $r8, implicit $r9, implicit $r10, implicit $r2, implicit-def $r1
; 32BIT-NEXT:     ADJCALLSTACKUP 92, 0, implicit-def dead $r1, implicit $r1

; CHECKASM-LABEL: .call_test_byval_mem3:

; ASM32BIT:       stwu 1, -112(1)
; ASM32BIT-DAG:   lwz [[REG:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM32BIT-DAG:   addi 3, 1, 56
; ASM32BIT-DAG:   addi 4, [[REG]], 24
; ASM32BIT-DAG:   li 5, 33
; ASM32BIT-NEXT:  bl .memcpy[PR]
; ASM32BIT-DAG:   lwz 5, 0([[REG]])
; ASM32BIT-DAG:   lwz 6, 4([[REG]])
; ASM32BIT-DAG:   lwz 7, 8([[REG]])
; ASM32BIT-DAG:   lwz 8, 12([[REG]])
; ASM32BIT-DAG:   lwz 9, 16([[REG]])
; ASM32BIT-DAG:   lwz 10, 20([[REG]])
; ASM32BIT:       bl .test_byval_mem3
; ASM32BIT:       addi 1, 1, 112

; The memcpy call was inlined in 64-bit so MIR test is redundant and omitted.
; ASM64BIT:       stdu 1, -128(1)
; ASM64BIT-DAG:   ld [[REG1:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64BIT-DAG:   ld [[REG2:[0-9]+]], 48([[REG1]])
; ASM64BIT-DAG:   std [[REG2]], 112(1)
; ASM64BIT-DAG:   lbz [[REG3:[0-9]+]], 56([[REG1]])
; ASM64BIT-DAG:   stb [[REG3]], 120(1)
; ASM64BIT-DAG:   ld 5, 0([[REG1]])
; ASM64BIT-DAG:   ld 6, 8([[REG1]])
; ASM64BIT-DAG:   ld 7, 16([[REG1]])
; ASM64BIT-DAG:   ld 8, 24([[REG1]])
; ASM64BIT-DAG:   ld 9, 32([[REG1]])
; ASM64BIT-DAG:   ld 10, 40([[REG1]])
; ASM64BIT:       bl .test_byval_mem3
; ASM64BIT:       addi 1, 1, 128

define void @test_byval_mem3(i32, float, %struct_S57* byval(%struct_S57) align 1 %s) {
entry:
  ret void
}


;CHECK-LABEL:  name:            test_byval_mem3

; 32BIT:      fixedStack:
; 32BIT-NEXT:   - { id: 0, type: default, offset: 32, size: 60, alignment: 16, stack-id: default,

; 32BIT:      bb.0.entry:
; 32BIT-NEXT:   liveins: $r5, $r6, $r7, $r8, $r9, $r10

; 32BIT-DAG:    %2:gprc = COPY $r5
; 32BIT-DAG:    %3:gprc = COPY $r6
; 32BIT-DAG:    %4:gprc = COPY $r7
; 32BIT-DAG:    %5:gprc = COPY $r8
; 32BIT-DAG:    %6:gprc = COPY $r9
; 32BIT-DAG:    %7:gprc = COPY $r10
; 32BIT-NEXT:   STW %2, 0, %fixed-stack.0 :: (store 4 into %fixed-stack.0
; 32BIT-DAG:    STW %3, 4, %fixed-stack.0 :: (store 4 into %fixed-stack.0 + 4
; 32BIT-DAG:    STW %4, 8, %fixed-stack.0 :: (store 4 into %fixed-stack.0 + 8
; 32BIT-DAG:    STW %5, 12, %fixed-stack.0 :: (store 4 into %fixed-stack.0 + 12
; 32BIT-DAG:    STW %6, 16, %fixed-stack.0 :: (store 4 into %fixed-stack.0 + 16
; 32BIT-DAG:    STW %7, 20, %fixed-stack.0 :: (store 4 into %fixed-stack.0 + 20
; 32BIT-NEXT:   BLR implicit $lr, implicit $rm

; 64BIT:      fixedStack:
; 64BIT-NEXT:   - { id: 0, type: default, offset: 64, size: 64, alignment: 16, stack-id: default,

; 64BIT:      bb.0.entry
; 64BIT-NEXT:   liveins: $x5, $x6, $x7, $x8, $x9, $x10

; 64BIT-DAG:    %2:g8rc = COPY $x5
; 64BIT-DAG:    %3:g8rc = COPY $x6
; 64BIT-DAG:    %4:g8rc = COPY $x7
; 64BIT-DAG:    %5:g8rc = COPY $x8
; 64BIT-DAG:    %6:g8rc = COPY $x9
; 64BIT-DAG:    %7:g8rc = COPY $x10
; 64BIT-NEXT:   STD %2, 0, %fixed-stack.0 :: (store 8 into %fixed-stack.0, align 16)
; 64BIT-DAG:    STD %3, 8, %fixed-stack.0 :: (store 8 into %fixed-stack.0 + 8)
; 64BIT-DAG:    STD %4, 16, %fixed-stack.0 :: (store 8 into %fixed-stack.0 + 16, align 16)
; 64BIT-DAG:    STD %5, 24, %fixed-stack.0 :: (store 8 into %fixed-stack.0 + 24)
; 64BIT-DAG:    STD %6, 32, %fixed-stack.0 :: (store 8 into %fixed-stack.0 + 32, align 16)
; 64BIT-DAG:    STD %7, 40, %fixed-stack.0 :: (store 8 into %fixed-stack.0 + 40)
; 64BIT-NEXT:   BLR8 implicit $lr8, implicit $rm

%struct_S31 = type { [31 x i8] }

@gS31 = external global %struct_S31, align 1

define void @call_test_byval_mem4() {
entry:
  call void @test_byval_mem4(i32 42, %struct_S31* byval(%struct_S31) align 1 @gS31, %struct_S256* byval(%struct_S256) align 1 @gS256)
  ret void
}


; CHECK-LABEL: name: call_test_byval_mem4

; CHECKASM-LABEL: .call_test_byval_mem4:

; Confirm the expected memcpy call is independent of the call to test_byval_mem4.
; 32BIT:          ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT:     %3:gprc = nuw ADDI $r1, 60
; 32BIT-NEXT:     %4:gprc = LWZtoc @gS256, $r2 :: (load 4 from got)
; 32BIT-NEXT:     %5:gprc = LI 256
; 32BIT-DAG:      $r3 = COPY %3
; 32BIT-DAG:      $r4 = COPY %4
; 32BIT-DAG:      $r5 = COPY %5
; 32BIT-NEXT:     BL_NOP &".memcpy[PR]", csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit $r5, implicit $r2, implicit-def $r1, implicit-def $r3
; 32BIT-NEXT:     ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT:          ADJCALLSTACKDOWN 316, 0, implicit-def dead $r1, implicit $r1
; 32BIT-DAG:      $r3 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r4 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r5 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r6 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r7 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r8 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r9 = COPY %{{[0-9]+}}
; 32BIT-DAG:      $r10 = COPY %{{[0-9]+}}
; 32BIT-NEXT:     BL_NOP <mcsymbol .test_byval_mem4>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r4, implicit $r5, implicit $r6, implicit $r7, implicit $r8, implicit $r9, implicit $r10, implicit $r2, implicit-def $r1
; 32BIT-NEXT:     ADJCALLSTACKUP 316, 0, implicit-def dead $r1, implicit $r1

; ASM32BIT:       stwu 1, -336(1)
; ASM32BIT-NEXT:  stw [[REG1:[0-9]+]], {{[0-9]+}}(1)
; ASM32BIT:       lwz [[REG1]], L..C{{[0-9]+}}(2)
; ASM32BIT-DAG:   lhz [[REG2:[0-9]+]], 28([[REG1]])
; ASM32BIT-DAG:   sth [[REG2]], 56(1)
; ASM32BIT-DAG:   lbz [[REG3:[0-9]+]], 30([[REG1]])
; ASM32BIT-DAG:   stb [[REG3]], 58(1)
; ASM32BIT-DAG:   addi 3, 1, 60
; ASM32BIT-DAG:   lwz 4, L..C{{[0-9]+}}(2)
; ASM32BIT-DAG:   li 5, 256
; ASM32BIT-NEXT:  bl .memcpy[PR]
; ASM32BIT-DAG:   lwz 4, 0([[REG1]])
; ASM32BIT-DAG:   lwz 5, 4([[REG1]])
; ASM32BIT-DAG:   lwz 6, 8([[REG1]])
; ASM32BIT-DAG:   lwz 7, 12([[REG1]])
; ASM32BIT-DAG:   lwz 8, 16([[REG1]])
; ASM32BIT-DAG:   lwz 9, 20([[REG1]])
; ASM32BIT-DAG:   lwz 10, 24([[REG1]])
; ASM32BIT:       bl .test_byval_mem4
; ASM32BIT:       addi 1, 1, 336

; Confirm the expected memcpy call is independent of the call to test_byval_mem4.
; 64BIT:          ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT:     %0:g8rc_and_g8rc_nox0 = LDtoc @gS256, $x2 :: (load 8 from got)
; 64BIT-NEXT:     %1:g8rc = nuw ADDI8 %0, 24
; 64BIT-NEXT:     %2:g8rc = nuw ADDI8 $x1, 112
; 64BIT-NEXT:     %3:g8rc = LI8 232
; 64BIT-DAG:      $x3 = COPY %2
; 64BIT-DAG:      $x4 = COPY %1
; 64BIT-DAG:      $x5 = COPY %3
; 64BIT-NEXT:     BL8_NOP &".memcpy[PR]", csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x4, implicit $x5, implicit $x2, implicit-def $r1, implicit-def $x3
; 64BIT-NEXT:     ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT:          ADJCALLSTACKDOWN 344, 0, implicit-def dead $r1, implicit $r1
; 64BIT-DAG:      $x3 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x4 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x5 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x6 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x7 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x8 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x9 = COPY %{{[0-9]+}}
; 64BIT-DAG:      $x10 = COPY %{{[0-9]+}}
; 64BIT-NEXT:     BL8_NOP <mcsymbol .test_byval_mem4>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x4, implicit $x5, implicit $x6, implicit $x7, implicit $x8, implicit $x9, implicit $x10, implicit $x2, implicit-def $r1
; 64BIT-NEXT:     ADJCALLSTACKUP 344, 0, implicit-def dead $r1, implicit $r1

; ASM64BIT:       stdu 1, -368(1)
; ASM64BIT-DAG:   ld [[REG1:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64BIT-DAG:   addi 3, 1, 112
; ASM64BIT-DAG:   addi 4, [[REG1]], 24
; ASM64BIT-DAG:   li 5, 232
; ASM64BIT-NEXT:  bl .memcpy[PR]
; ASM64BIT-DAG:   ld [[REG2:[0-9]+]], L..C{{[0-9]+}}(2)
; ASM64BIT-DAG:   ld 4, 0([[REG2]])
; ASM64BIT-DAG:   ld 5, 8([[REG2]])
; ASM64BIT-DAG:   ld 6, 16([[REG2]])
; ASM64BIT-DAG:   lwz [[REG3:[0-9]+]], 24([[REG2]])
; ASM64BIT-DAG:   lhz [[REG4:[0-9]+]], 28([[REG2]])
; ASM64BIT-DAG:   lbz 7, 30([[REG2]])
; ASM64BIT-DAG:   rlwinm 7, 7, 8, 16, 23
; ASM64BIT-DAG:   rlwimi 7, [[REG4]], 16, 0, 15
; ASM64BIT-DAG:   rldimi 7, [[REG3]], 32, 0
; ASM64BIT-DAG:   ld 8, 0([[REG1]])
; ASM64BIT-DAG:   ld 9, 8([[REG1]])
; ASM64BIT-DAG:   ld 10, 16([[REG1]])
; ASM64BIT:       bl .test_byval_mem4
; ASM64BIT:       addi 1, 1, 368

define void @test_byval_mem4(i32, %struct_S31* byval(%struct_S31) align 1, %struct_S256* byval(%struct_S256) align 1 %s) {
entry:
  ret void
}

; CHECK-LABEL:    name:            test_byval_mem4

; 32BIT:          fixedStack:
; 32BIT:            - { id: 0, type: default, offset: 60, size: 256, alignment: 4, stack-id: default,
; 32BIT:            - { id: 1, type: default, offset: 28, size: 32, alignment: 4, stack-id: default,
; 32BIT:          stack:           []

; 32BIT:          bb.0.entry:
; 32BIT-NEXT:       liveins: $r4, $r5, $r6, $r7, $r8, $r9, $r10

; 32BIT-DAG:      %1:gprc = COPY $r4
; 32BIT-DAG:      %2:gprc = COPY $r5
; 32BIT-DAG:      %3:gprc = COPY $r6
; 32BIT-DAG:      %4:gprc = COPY $r7
; 32BIT-DAG:      %5:gprc = COPY $r8
; 32BIT-DAG:      %6:gprc = COPY $r9
; 32BIT-DAG:      %7:gprc = COPY $r10
; 32BIT-NEXT:     STW %1, 0, %fixed-stack.1 :: (store 4 into %fixed-stack.1
; 32BIT-DAG:      STW %2, 4, %fixed-stack.1 :: (store 4 into %fixed-stack.1 + 4
; 32BIT-DAG:      STW %3, 8, %fixed-stack.1 :: (store 4 into %fixed-stack.1 + 8
; 32BIT-DAG:      STW %4, 12, %fixed-stack.1 :: (store 4 into %fixed-stack.1 + 12
; 32BIT-DAG:      STW %5, 16, %fixed-stack.1 :: (store 4 into %fixed-stack.1 + 16
; 32BIT-DAG:      STW %6, 20, %fixed-stack.1 :: (store 4 into %fixed-stack.1 + 20
; 32BIT-DAG:      STW %7, 24, %fixed-stack.1 :: (store 4 into %fixed-stack.1 + 24
; 32BIT-NEXT:     BLR implicit $lr, implicit $rm

; 64BIT:          fixedStack:
; 64BIT:            - { id: 0, type: default, offset: 88, size: 256, alignment: 8, stack-id: default,
; 64BIT:            - { id: 1, type: default, offset: 56, size: 32, alignment: 8, stack-id: default,
; 64BIT:          stack:           []

; 64BIT:          bb.0.entry:
; 64BIT-NEXT:       liveins: $x4, $x5, $x6, $x7, $x8, $x9, $x10

; 64BIT-DAG:      %1:g8rc = COPY $x4
; 64BIT-DAG:      %2:g8rc = COPY $x5
; 64BIT-DAG:      %3:g8rc = COPY $x6
; 64BIT-DAG:      %4:g8rc = COPY $x7
; 64BIT-DAG:      %5:g8rc = COPY $x8
; 64BIT-DAG:      %6:g8rc = COPY $x9
; 64BIT-DAG:      %7:g8rc = COPY $x10
; 64BIT-NEXT:     STD %1, 0, %fixed-stack.1 :: (store 8 into %fixed-stack.1
; 64BIT-DAG:      STD %2, 8, %fixed-stack.1 :: (store 8 into %fixed-stack.1 + 8
; 64BIT-DAG:      STD %3, 16, %fixed-stack.1 :: (store 8 into %fixed-stack.1 + 16
; 64BIT-DAG:      STD %4, 24, %fixed-stack.1 :: (store 8 into %fixed-stack.1 + 24
; 64BIT-DAG:      STD %5, 0, %fixed-stack.0 :: (store 8 into %fixed-stack.0
; 64BIT-DAG:      STD %6, 8, %fixed-stack.0 :: (store 8 into %fixed-stack.0 + 8
; 64BIT-DAG:      STD %7, 16, %fixed-stack.0 :: (store 8 into %fixed-stack.0 + 16
; 64BIT-NEXT:     BLR8 implicit $lr8, implicit $rm

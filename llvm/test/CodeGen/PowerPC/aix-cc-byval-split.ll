; RUN: llc -mtriple powerpc-ibm-aix-xcoff -stop-after=machine-cp \
; RUN:   -mcpu=pwr4 -mattr=-altivec -verify-machineinstrs 2>&1 < %s | \
; RUN:    FileCheck --check-prefix=CHECK32 %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp \
; RUN:   -mcpu=pwr4 -mattr=-altivec -verify-machineinstrs 2>&1 < %s | \
; RUN:   FileCheck --check-prefix=CHECK64  %s

%struct.Spill = type { [12 x i64 ] }
@GS = external global %struct.Spill, align 4

define i64 @test(ptr byval(%struct.Spill) align 4 %s) {
entry:
  %arrayidx_a = getelementptr inbounds %struct.Spill, ptr %s, i32 0, i32 0, i32 2
  %arrayidx_b = getelementptr inbounds %struct.Spill, ptr %s, i32 0, i32 0, i32 10
  %a = load i64, ptr %arrayidx_a
  %b = load i64, ptr %arrayidx_b
  %add = add i64 %a, %b
  ret i64 %add
}

; CHECK32:  name:            test
; CHECK32:  liveins:
; CHECK32:    - { reg: '$r3', virtual-reg: '' }
; CHECK32:    - { reg: '$r4', virtual-reg: '' }
; CHECK32:    - { reg: '$r5', virtual-reg: '' }
; CHECK32:    - { reg: '$r6', virtual-reg: '' }
; CHECK32:    - { reg: '$r7', virtual-reg: '' }
; CHECK32:    - { reg: '$r8', virtual-reg: '' }
; CHECK32:    - { reg: '$r9', virtual-reg: '' }
; CHECK32:    - { reg: '$r10', virtual-reg: '' }
; CHECK32:  fixedStack:
; CHECK32:    - { id: 0, type: default, offset: 24, size: 96, alignment: 8, stack-id: default,
; CHECK32:  stack:           []

; CHECK32:      bb.0.entry:
; CHECK32-NEXT:   liveins: $r3, $r4, $r5, $r6, $r7, $r8, $r9, $r10

; CHECK32:     renamable $r[[REG1:[0-9]+]] = LWZ 84, %fixed-stack.0
; CHECK32-DAG: STW killed renamable $r3, 0, %fixed-stack.0 :: (store (s32) into %fixed-stack.0
; CHECK32-DAG: STW killed renamable $r4, 4, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 4
; CHECK32:     renamable $r[[REG2:[0-9]+]] = LWZ 80, %fixed-stack.0
; CHECK32-DAG: STW killed renamable $r5, 8, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 8
; CHECK32-DAG: STW killed renamable $r6, 12, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 12
; CHECK32-DAG: STW        renamable $r7, 16, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 16
; CHECK32-DAG: STW        renamable $r8, 20, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 20
; CHECK32-DAG: STW killed renamable $r9, 24, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 24
; CHECK32:     renamable $r4 = ADDC killed renamable $r8, killed renamable $r[[REG1]], implicit-def $carry
; CHECK32:     renamable $r3 = ADDE killed renamable $r7, killed renamable $r[[REG2]], implicit-def dead $carry, implicit killed $carry
; CHECK32      STW killed renamable $r10, 28, %fixed-stack.0 :: (store (s32) into %fixed-stack.0 + 28
; CHECK32:     BLR implicit $lr, implicit $rm, implicit $r3, implicit $r4


; CHECK64:  name:            test
; CHECK64:  liveins:
; CHECK64:    - { reg: '$x3', virtual-reg: '' }
; CHECK64:    - { reg: '$x4', virtual-reg: '' }
; CHECK64:    - { reg: '$x5', virtual-reg: '' }
; CHECK64:    - { reg: '$x6', virtual-reg: '' }
; CHECK64:    - { reg: '$x7', virtual-reg: '' }
; CHECK64:    - { reg: '$x8', virtual-reg: '' }
; CHECK64:    - { reg: '$x9', virtual-reg: '' }
; CHECK64:    - { reg: '$x10', virtual-reg: '' }
; CHECK64:  fixedStack:
; CHECK64:    - { id: 0, type: default, offset: 48, size: 96, alignment: 16, stack-id: default,
; CHECK64:  stack:           []

; CHECK64:  bb.0.entry:
; CHECK64:    liveins: $x3, $x4, $x5, $x6, $x7, $x8, $x9, $x10

; CHECK64: renamable $x[[REG1:[0-9]+]] = LD 80, %fixed-stack.0
; CHECK64: STD killed renamable $x3, 0, %fixed-stack.0 :: (store (s64) into %fixed-stack.0
; CHECK64: STD killed renamable $x4, 8, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 8
; CHECK64: STD renamable        $x5, 16, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 16
; CHECK64: STD killed renamable $x6, 24, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 24
; CHECK64: STD killed renamable $x7, 32, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 32
; CHECK64: STD killed renamable $x8, 40, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 40
; CHECK64: STD killed renamable $x9, 48, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 48
; CHECK64: renamable $x3 = ADD8 killed renamable $x5, killed renamable $x[[REG1]]
; CHECK64: STD killed renamable $x10, 56, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 56
; CHECK64: BLR8 implicit $lr8, implicit $rm, implicit $x3

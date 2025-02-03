; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu --verify-machineinstrs \
; RUN: -stop-after=prologepilog < %s | FileCheck \
; RUN: --check-prefixes=CHECK,CHECK64,ELFV2 %s

; RUN: llc -mtriple=powerpc64-unknown-aix-xcoff -mcpu=pwr4 \
; RUN: --verify-machineinstrs --mattr=-altivec -stop-after=prologepilog < %s | \
; RUN: FileCheck --check-prefixes=CHECK,CHECK64,V1ANDAIX  %s

; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 --verify-machineinstrs \
; RUN: -stop-after=prologepilog < %s | FileCheck \
; RUN: --check-prefixes=CHECK,CHECK64,V1ANDAIX %s

; RUN: llc -mtriple=powerpc-unknown-aix-xcoff -mcpu=pwr4 \
; RUN: --verify-machineinstrs --mattr=-altivec -stop-after=prologepilog < %s | \
; RUN: FileCheck --check-prefixes=CHECK,CHECK32  %s

define dso_local signext i32 @test(i32 signext %n) {
entry:
  %conv = sext i32 %n to i64
  %0 = alloca double, i64 %conv, align 16
  tail call void asm sideeffect "", "~{cr2}"()
  %call = call signext i32 @do_something(ptr nonnull %0)
  ret i32 %call
}

declare signext i32 @do_something(ptr)

; CHECK: name:            test
; CHECK: alignment:       16
; CHECK: liveins:
; CHECK64:   - { reg: '$x3', virtual-reg: '' }
; CHECK32:   - { reg: '$r3', virtual-reg: '' }

; ELFV2:    stackSize:       48
; V1ANDAIX: stackSize:       128
; CHECK32:  stackSize:       80

; ELFV2:    maxCallFrameSize: 32
; V1ANDAIX: maxCallFrameSize: 112
; CHECK32:  maxCallFrameSize: 64

; CHECK64:      fixedStack:
; CHECK64-NEXT:   - { id: 0, type: default, offset: 8, size: 4, alignment: 8, stack-id: default,
; CHECK64-NEXT:       isImmutable: true, isAliased: false, callee-saved-register: '$cr2',
; CHECK64-NEXT:       callee-saved-restored: true, debug-info-variable: '', debug-info-expression: '',
; CHECK64-NEXT:       debug-info-location: '' }
; CHECK64-NEXT:   - { id: 1, type: default, offset: -8, size: 8, alignment: 8, stack-id: default,
; CHECK64-NEXT:       isImmutable: true, isAliased: false, callee-saved-register: '', callee-saved-restored: true,
; CHECK64-NEXT:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }

; CHECK64-NEXT: stack:
; CHECK64-NEXT:   - { id: 0, name: '', type: variable-sized, offset: -8, alignment: 1, 
; CHECK64-NEXT:       stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK64-NEXT:       local-offset: 0, debug-info-variable: '', debug-info-expression: '',
; CHECK64-NEXT:       debug-info-location: '' }
; CHECK64-NEXT:   - { id: 1, name: '', type: spill-slot, offset: -16, size: 8, alignment: 8,
; CHECK64-NEXT:       stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK64-NEXT:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }


; CHECK32:      fixedStack:
; CHECK32-NEXT:   - { id: 0, type: default, offset: 4, size: 4, alignment: 4, stack-id: default,
; CHECK32-NEXT:       isImmutable: true, isAliased: false, callee-saved-register: '$cr2',
; CHECK32-NEXT:       callee-saved-restored: true, debug-info-variable: '', debug-info-expression: '',
; CHECK32-NEXT:       debug-info-location: '' }
; CHECK32-NEXT:   - { id: 1, type: default, offset: -4, size: 4, alignment: 4, stack-id: default,
; CHECK32-NEXT:       isImmutable: true, isAliased: false, callee-saved-register: '', callee-saved-restored: true,
; CHECK32-NEXT:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }

; CHECK32-NEXT: stack:
; CHECK32-NEXT:   - { id: 0, name: '', type: variable-sized, offset: -4, alignment: 1, 
; CHECK32-NEXT:       stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK32-NEXT:       local-offset: 0, debug-info-variable: '', debug-info-expression: '',
; CHECK32-NEXT:       debug-info-location: '' }
; CHECK32-NEXT:   - { id: 1, name: '', type: spill-slot, offset: -8, size: 4, alignment: 4,
; CHECK32-NEXT:       stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK32-NEXT:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }


; CHECK64:      bb.0.entry:
; CHECK64-NEXT:  liveins: $x3, $cr2

; Prologue:
; CHECK64:         $x0 = MFLR8 implicit $lr8
; ELFV2-NEXT:      $x12 = MFOCRF8 killed $cr2
; V1ANDAIX-NEXT:   $x12 = MFCR8 implicit killed $cr2
; CHECK64-DAG:     STD $x31, -8, $x1
; CHECK64-DAG:     STW8 killed $x12, 8, $x1

; ELFV2-NEXT:      $x1 = STDU $x1, -48, $x1
; ELFV2-NEXT:      STD killed $x0, 64, $x1
; V1ANDAIX-NEXT:   x1 = STDU $x1, -128, $x1
; V1ANDAIX-NEXT:   STD killed $x0, 144, $x1

; CHECK64:         $x31 = OR8 $x1, $x1

; ELFV2:    $[[ORIGSP:x[0-9]+]] = ADDI8 $x31, 48
; V1ANDAIX: $[[ORIGSP:x[0-9]+]] = ADDI8 $x31, 128
; CHECK64:  $x1 = STDUX killed $[[ORIGSP]], $x1, killed $x{{[0-9]}}
; CHECK64:  INLINEASM {{.*}} early-clobber $cr2
; CHECK64:  BL8_NOP


; Epilogue:
; CHECK64:       $x1 = LD 0, $x1
; CHECK64-DAG:   $x0 = LD 16, $x1
; CHECK64-DAG:   $x12 = LWZ8 8, $x1
; CHECK64-DAG:   $x31 = LD -8, $x1
; CHECK64:       $cr2 = MTOCRF8 killed $x12
; CHECK64-NEXT:  MTLR8 $x0, implicit-def $lr8
; CHECK64-NEXT:  BLR8 implicit $lr8, implicit $rm, implicit $x3

; CHECK32:       bb.0.entry:
; CHECK32-NEXT:    liveins: $r3, $cr2

; Prologue:
; CHECK32:       $r0 = MFLR implicit $lr
; CHECK32-NEXT:  $r12 = MFCR implicit killed $cr2
; CHECK32-DAG:   STW $r31, -4, $r1
; CHECK32-DAG:   STW killed $r12, 4, $r1
; CHECK32:       $r1 = STWU $r1, -80, $r1
; CHECK32-NEXT:  STW killed $r0, 88, $r1

; CHECK32:       $r31 = OR $r1, $r1
; CHECK32:       $[[ORIGSP:r[0-9]+]] = ADDI $r31, 80
; CHECK32:       $r1 = STWUX killed $[[ORIGSP]], $r1, killed $r{{[0-9]}}
; CHECK32:       INLINEASM {{.*}} early-clobber $cr2
; CHECK32:       BL_NOP

; Epilogue:
; CHECK32:       $r1 = LWZ 0, $r1
; CHECK32-DAG:   $r0 = LWZ 8, $r1
; CHECK32-DAG:   $r12 = LWZ 4, $r1
; CHECK32-DAG:   $r31 = LWZ -4, $r1
; CHECK32:       $cr2 = MTOCRF killed $r12
; CHECK32-NEXT:  MTLR $r0, implicit-def $lr
; CHECK32-NEXT:  BLR implicit $lr, implicit $rm, implicit $r3

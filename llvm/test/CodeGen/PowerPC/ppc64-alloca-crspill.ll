; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu --verify-machineinstrs \
; RUN: -stop-after=prologepilog < %s | FileCheck %s

define dso_local signext i32 @test(i32 signext %n) {
entry:
  %conv = sext i32 %n to i64
  %0 = alloca double, i64 %conv, align 16
  tail call void asm sideeffect "", "~{cr2}"()
  %call = call signext i32 @do_something(double* nonnull %0)
  ret i32 %call
}

declare signext i32 @do_something(double*)

; CHECK: name:            test
; CHECK: alignment:       16
; CHECK: liveins:
; CHECK:   - { reg: '$x3', virtual-reg: '' }
; CHECK: stackSize:       48
; CHECK: maxCallFrameSize: 32

; CHECK:      fixedStack:
; CHECK-NEXT:   - { id: 0, type: default, offset: 8, size: 4, alignment: 8, stack-id: default,
; CHECK-NEXT:       isImmutable: true, isAliased: false, callee-saved-register: '$cr2',
; CHECK-NEXT:       callee-saved-restored: true, debug-info-variable: '', debug-info-expression: '',
; CHECK-NEXT:       debug-info-location: '' }
; CHECK-NEXT:   - { id: 1, type: default, offset: -8, size: 8, alignment: 8, stack-id: default,
; CHECK-NEXT:       isImmutable: true, isAliased: false, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }

; CHECK-NEXT: stack:
; CHECK-NEXT:   - { id: 0, name: '<unnamed alloca>', type: variable-sized, offset: -8,
; CHECK-NEXT:       alignment: 1, stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT:       local-offset: 0, debug-info-variable: '', debug-info-expression: '',
; CHECK-NEXT:       debug-info-location: '' }
; CHECK-NEXT:   - { id: 1, name: '', type: default, offset: -16, size: 8, alignment: 8,
; CHECK-NEXT:       stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }

; CHECK:      bb.0.entry:
; CHECK-NEXT:  liveins: $x3, $cr2

; Prologue:
; CHECK:       $x0 = MFLR8 implicit $lr8
; CHECK-NEXT:  $x12 = MFOCRF8 killed $cr2
; CHECK-DAG:   STD $x31, -8, $x1
; CHECK-DAG:   STD killed $x0, 16, $x1
; CHECK-DAG:   STW8 killed $x12, 8, $x1
; CHECK-NEXT:  $x1 = STDU $x1, -48, $x1
; CHECK:       $x31 = OR8 $x1, $x1

; CHECK: $[[ORIGSP:x[0-9]+]] = ADDI8 $x31, 48
; CHECK: $x1 = STDUX killed $[[ORIGSP]], $x1, killed $x{{[0-9]}}
; CHECK: INLINEASM {{.*}} early-clobber $cr2
; CHECK: BL8_NOP @do_something


; Epilogue:
; CHECK:       $x1 = LD 0, $x1
; CHECK-DAG:   $x0 = LD 16, $x1
; CHECK-DAG:   $x12 = LWZ8 8, $x1
; CHECK-DAG:   $x31 = LD -8, $x1
; CHECK:       $cr2 = MTOCRF8 killed $x12
; CHECK-NEXT:  MTLR8 $x0, implicit-def $lr8
; CHECK-NEXT:  BLR8 implicit $lr8, implicit $rm, implicit $x3


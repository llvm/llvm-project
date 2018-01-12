; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
; The instructions addis,addi, bl are used to calculate the address of TLS
; thread local variables. These TLS access code sequences are generated
; repeatedly every time the thread local variable is accessed. By communicating
; to Machine CSE that X2 is guaranteed to have the same value within the same
; function call (so called Caller Preserved Physical Register), the redudant
; TLS access code sequences are cleaned up.

%"struct.CC::TT" = type { i64, i32 }
%class.CC = type { %struct.SS }
%struct.SS = type { void ()* }

@_ZN2CC2ccE = external thread_local global %"struct.CC::TT", align 8

define noalias i8* @_ZN2CC3funEv(%class.CC* %this) {
; CHECK-LABEL: _ZN2CC3funEv:
; CHECK:    mflr 0
; CHECK-NEXT:    std 0, 16(1)
; CHECK-NEXT:    stdu 1, -48(1)
; CHECK-NEXT:    .cfi_def_cfa_offset 48
; CHECK-NEXT:    .cfi_offset lr, 16
; CHECK-NEXT:    .cfi_offset r30, -16
; CHECK-NEXT:    std 30, 32(1)
; CHECK-NEXT:    mr 30, 3
; CHECK-NEXT:    ld 12, 0(30)
; CHECK-NEXT:    std 2, 24(1)
; CHECK-NEXT:    mtctr 12
; CHECK-NEXT:    bctrl
; CHECK-NEXT:    ld 2, 24(1)
; CHECK-NEXT:    addis 3, 2, _ZN2CC2ccE@got@tlsgd@ha
; CHECK-NEXT:    addi 3, 3, _ZN2CC2ccE@got@tlsgd@l
; CHECK-NEXT:    bl __tls_get_addr(_ZN2CC2ccE@tlsgd)
; CHECK-NEXT:    nop
; CHECK-NEXT:    ld 4, 0(3)
; CHECK-NEXT:    cmpldi 4, 0
; CHECK-NEXT:    beq 0, .LBB0_2
; CHECK:    addi 4, 3, 8
; CHECK-NEXT:    mr 3, 30
; CHECK-NEXT:    bl _ZN2CC3barEPi
; CHECK-NEXT:    nop
; CHECK:    ld 30, 32(1)
; CHECK-NEXT:    li 3, 0
; CHECK-NEXT:    addi 1, 1, 48
; CHECK-NEXT:    ld 0, 16(1)
; CHECK-NEXT:    mtlr 0
; CHECK-NEXT:    blr
entry:
  %foo = getelementptr inbounds %class.CC, %class.CC* %this, i64 0, i32 0, i32 0
  %0 = load void ()*, void ()** %foo, align 8
  tail call void %0()
  %1 = load i64, i64* getelementptr inbounds (%"struct.CC::TT", %"struct.CC::TT"* @_ZN2CC2ccE, i64 0, i32 0)
  %tobool = icmp eq i64 %1, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  tail call void @_ZN2CC3barEPi(%class.CC* nonnull %this, i32* getelementptr inbounds (%"struct.CC::TT", %"struct.CC::TT"* @_ZN2CC2ccE, i64 0, i32 1))
  br label %if.end

if.end:
  ret i8* null
}

declare void @_ZN2CC3barEPi(%class.CC*, i32*)

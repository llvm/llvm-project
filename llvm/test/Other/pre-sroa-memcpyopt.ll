; RUN: opt -passes='default<O2>' -debug-pass-manager -disable-output %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ORDER
; RUN: opt -passes='default<O2>' -print-before=sroa -disable-output %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=IR

; ORDER: Running pass: EntryExitInstrumenterPass
; ORDER-NEXT: Running pass: LowerExpectIntrinsicPass
; ORDER-NEXT: Running pass: SimplifyCFGPass
; ORDER-NOT: Running pass: SROAPass
; ORDER: Running pass: MemCpyOptPass
; ORDER: Running pass: SROAPass

; IR-LABEL: *** IR Dump Before SROAPass on copy ***
; IR: define void @copy
; IR: %tmp = alloca [268435455 x i8]
; IR-NOT: load [268435455 x i8]
; IR-NOT: store [268435455 x i8]
; IR-COUNT-2: call void @llvm.memcpy
; IR: ret void

target triple = "x86_64-unknown-linux-gnu"

%Large = type [268435455 x i8]

define void @copy(ptr noalias %src, ptr noalias %dst) {
entry:
  %tmp = alloca %Large, align 1
  %value = load %Large, ptr %src, align 1
  store %Large %value, ptr %tmp, align 1
  %result = load %Large, ptr %tmp, align 1
  store %Large %result, ptr %dst, align 1
  ret void
}

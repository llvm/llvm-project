; RUN: llc -mcpu=cyclone -debug-only=misched < %s 2>&1 | FileCheck %s

; REQUIRES: asserts

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios7.0.0"

define void @caller2(i8* %a0, i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9) {
entry:
  tail call void @callee2(i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a0)
  ret void
}

declare void @callee2(i8*, i8*, i8*, i8*, i8*,
                      i8*, i8*, i8*, i8*, i8*)

; Make sure there is a dependence between the load and store to the same stack
; location during a tail call. Tail calls clobber the incoming argument area and
; therefore it is not safe to assume argument locations are invariant.
; PR23459 has a test case that we where miscompiling because of this at the
; time.

; CHECK: Frame Objects
; CHECK:  fi#-4: {{.*}} fixed, at location [SP+8]
; CHECK:  fi#-3: {{.*}} fixed, at location [SP]
; CHECK:  fi#-2: {{.*}} fixed, at location [SP+8]
; CHECK:  fi#-1: {{.*}} fixed, at location [SP]

; CHECK:  [[VRA:%vreg.*]]<def> = LDRXui <fi#-1>
; CHECK:  [[VRB:%vreg.*]]<def> = LDRXui <fi#-2>
; CHECK:  STRXui %vreg{{.*}}, <fi#-4>
; CHECK:  STRXui [[VRB]], <fi#-3>

; Make sure that there is an dependence edge between fi#-2 and fi#-4.
; Without this edge the scheduler would be free to move the store accross the load.

; CHECK: SU({{.*}}):   [[VRB]]<def> = LDRXui <fi#-2>
; CHECK-NOT: SU
; CHECK:  Successors:
; CHECK:   ch  SU([[DEPSTOREB:.*]]): Latency=0
; CHECK:   ch  SU([[DEPSTOREA:.*]]): Latency=0

; CHECK: SU([[DEPSTOREA]]):   STRXui %vreg{{.*}}, <fi#-4>
; CHECK: SU([[DEPSTOREB]]):   STRXui %vreg{{.*}}, <fi#-3>

; RUN: llc -mcpu=cyclone -debug-only=machine-scheduler < %s 2>&1 | FileCheck %s --check-prefixes=COMMON,SDAG
; RUN: llc -mcpu=cyclone -global-isel -debug-only=machine-scheduler < %s 2>&1 | FileCheck %s --check-prefixes=COMMON,GISEL

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

; COMMON: Frame Objects
; COMMON:  fi#-4: {{.*}} fixed, at location [SP+8]
; COMMON:  fi#-3: {{.*}} fixed, at location [SP]
; COMMON:  fi#-2: {{.*}} fixed, at location [SP+8]
; COMMON:  fi#-1: {{.*}} fixed, at location [SP]

; The order that these appear in differes in GISel than SDAG, but the
; dependency relationship still holds.
; COMMON:  [[VRA:%.*]]:gpr64 = LDRXui %fixed-stack.3
; COMMON:  [[VRB:%.*]]:gpr64 = LDRXui %fixed-stack.2
; SDAG:  STRXui %{{.*}}, %fixed-stack.0
; SDAG:  STRXui [[VRB]]{{[^,]*}}, %fixed-stack.1
; GISEL:  STRXui [[VRB]]{{[^,]*}}, %fixed-stack.1
; GISEL:  STRXui %{{.*}}, %fixed-stack.0

; Make sure that there is an dependence edge between fi#-2 and fi#-4.
; Without this edge the scheduler would be free to move the store accross the load.

; COMMON: {{^SU(.*)}}:   [[VRB]]:gpr64 = LDRXui %fixed-stack.2
; COMMON-NOT: {{^SU(.*)}}:
; COMMON:  Successors:
; COMMON:   SU([[DEPSTOREB:.*]]): Ord  Latency=0
; COMMON:   SU([[DEPSTOREA:.*]]): Ord  Latency=0

; GlobalISel outputs DEPSTOREB before DEPSTOREA, but the dependency relationship
; still holds.
; SDAG: SU([[DEPSTOREA]]):   STRXui %{{.*}}, %fixed-stack.0
; SDAG: SU([[DEPSTOREB]]):   STRXui %{{.*}}, %fixed-stack.1

; GISEL: SU([[DEPSTOREB]]):   STRXui %{{.*}}, %fixed-stack.0
; GISEL: SU([[DEPSTOREA]]):   STRXui %{{.*}}, %fixed-stack.1

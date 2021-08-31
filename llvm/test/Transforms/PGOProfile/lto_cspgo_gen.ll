; REQUIRES: x86-registered-target

; RUN: opt -passes='thinlto-pre-link<O2>' --cs-profilegen-file=alloc -cspgo-kind=cspgo-instr-gen-pipeline -module-summary %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=IRPGOPRE

;; Symbol __llvm_profile_filename and __llvm_profile_raw_version are non-prevailing here.
; RUN: llvm-lto2 run -lto-cspgo-profile-file=alloc -lto-cspgo-gen -save-temps -o %t %t.bc \
; RUN:   -r=%t.bc,f,px \
; RUN:   -r=%t.bc,__llvm_profile_filename,x \
; RUN:   -r=%t.bc,__llvm_profile_raw_version,x
; RUN: llvm-dis %t.0.0.preopt.bc -o - | FileCheck %s --check-prefix=IRPGOBE

;; Before LTO, we should have the __llvm_profile_raw_version definition.
; IRPGOPRE: @__llvm_profile_raw_version = constant i64

;; Non-prevailing __llvm_profile_raw_version is discarded by LTO. Ensure the
;; declaration is retained.
; IRPGOBE: @__llvm_profile_raw_version = external constant i64

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$f = comdat any

; Function Attrs: nofree norecurse nosync nounwind readnone uwtable willreturn mustprogress
define i32 @f() {
entry:
  ret i32 1
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ThinLTO", i32 0}

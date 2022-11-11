; RUN: llvm-profdata merge %S/Inputs/multiple_hash_profile.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -pgo-instr-old-cfg-hashing=true -S | FileCheck -check-prefix=CHECKOLDHASH %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_Z3fooi = comdat any

@g2 = local_unnamed_addr global ptr null, align 8

define i32 @_Z3bari(i32 %i) {
entry:
  %cmp = icmp sgt i32 %i, 2
  %mul = select i1 %cmp, i32 1, i32 %i
  %retval.0 = mul nsw i32 %mul, %i
  ret i32 %retval.0
}

define void @_Z4m2f1v() {
entry:
  store ptr @_Z3fooi, ptr @g2, align 8
  ret void
}

define linkonce_odr i32 @_Z3fooi(i32 %i) comdat {
entry:
  %cmp.i = icmp sgt i32 %i, 2
  %mul.i = select i1 %cmp.i, i32 1, i32 %i
; CHECK: %mul.i = select i1 %cmp.i, i32 1, i32 %i
; CHECK-SAME: !prof ![[BW:[0-9]+]]
; CHECK: ![[BW]] = !{!"branch_weights", i32 12, i32 6}
; CHECKOLDHASH: %mul.i = select i1 %cmp.i, i32 1, i32 %i
; CHECKOLDHASH-SAME: !prof ![[BW:[0-9]+]]
; CHECKOLDHASH: ![[BW]] = !{!"branch_weights", i32 6, i32 12}
  %retval.0.i = mul nsw i32 %mul.i, %i
  ret i32 %retval.0.i
}



; Verify that when the contiguity-guard flag is enabled and the parent loop
; has a constant byte stride that matches the inner-loop store/load shape,
; the YAML pass-remarks-output records Name: ContiguityGuardedMemset/Memcpy
; and ContiguityGuarded: 'true'. This complements memset-debugify-remarks.ll /
; memcpy-debugify-remarks.ll, which exercise the default (flag-off) path.

; RUN: rm -f %t.memset.yaml %t.memcpy.yaml
; RUN: opt -passes=loop-idiom -loop-idiom-enable-memset-cont-check \
; RUN:   -pass-remarks-output=%t.memset.yaml -verify-each -verify-dom-info \
; RUN:   -verify-loop-info < %s -S -o /dev/null
; RUN: FileCheck --input-file=%t.memset.yaml %s --check-prefix=YAML-MEMSET
; RUN: opt -passes=loop-idiom -loop-idiom-enable-memcpy-cont-check \
; RUN:   -pass-remarks-output=%t.memcpy.yaml -verify-each -verify-dom-info \
; RUN:   -verify-loop-info < %s -S -o /dev/null
; RUN: FileCheck --input-file=%t.memcpy.yaml %s --check-prefix=YAML-MEMCPY
; RUN: opt -passes=loop-idiom -loop-idiom-enable-memset-cont-check -pass-remarks=loop-idiom \
; RUN:   < %s -S -o /dev/null 2>&1 | FileCheck %s --check-prefix=RPASS-MEMSET
; RUN: opt -passes=loop-idiom -loop-idiom-enable-memcpy-cont-check -pass-remarks=loop-idiom \
; RUN:   < %s -S -o /dev/null 2>&1 | FileCheck %s --check-prefix=RPASS-MEMCPY
; RUN: opt -passes=loop-idiom -loop-idiom-enable-memset-cont-check \
; RUN:   -debug-only=loop-idiom < %s -S -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DEBUG-MEMSET
; RUN: opt -passes=loop-idiom -loop-idiom-enable-memcpy-cont-check \
; RUN:   -debug-only=loop-idiom < %s -S -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DEBUG-MEMCPY

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; YAML-MEMSET:      --- !Passed
; YAML-MEMSET:      Pass:            loop-idiom
; YAML-MEMSET:      Name:            ContiguityGuardedMemset
; YAML-MEMSET:      Function:        memset_cont_specialized
; YAML-MEMSET:        - NewFunction:     llvm.memset
; YAML-MEMSET:        - String:          '() intrinsic'
; YAML-MEMSET-NEXT:   - ContiguityGuarded: 'true'

; RPASS-MEMSET: Formed contiguity-guarded memset
; RPASS-MEMSET-NOT: contiguity-guarded memcpy

; DEBUG-MEMSET: contiguity-guarded memset
; DEBUG-MEMSET-NOT: contiguity-guarded memcpy

define void @memset_cont_specialized(ptr noalias nocapture writeonly %arr, ptr noalias nocapture readonly %ubnd) local_unnamed_addr {
L.entry:
  %0 = load i32, ptr %ubnd, align 4
  %1 = icmp slt i32 %0, 1
  br i1 %1, label %L.exit, label %L.outer.preheader

L.outer.preheader:
  %2 = sext i32 %0 to i64
  %base = getelementptr i8, ptr %arr, i64 -20
  br label %L.outer

L.outer:
  %outer.iv = phi i64 [ 100, %L.outer.preheader ], [ %outer.next, %L.outer.latch ]
  %ia = phi i64 [ 1, %L.outer.preheader ], [ %ia.next, %L.outer.latch ]
  %row.off = shl nsw i64 %ia, 2
  br label %L.inner

L.inner:
  %inner.iv = phi i64 [ %2, %L.outer ], [ %inner.dec, %L.inner ]
  %ib = phi i64 [ 1, %L.outer ], [ %ib.next, %L.inner ]
  %sum = add nuw nsw i64 %ib, %row.off
  %p = getelementptr i32, ptr %base, i64 %sum
  store i32 0, ptr %p, align 4
  %ib.next = add nuw nsw i64 %ib, 1
  %inner.dec = add nsw i64 %inner.iv, -1
  %inner.alive = icmp sgt i64 %inner.iv, 1
  br i1 %inner.alive, label %L.inner, label %L.outer.latch

L.outer.latch:
  %ia.next = add nuw nsw i64 %ia, 1
  %outer.next = add nsw i64 %outer.iv, -1
  %outer.alive = icmp sgt i64 %outer.iv, 1
  br i1 %outer.alive, label %L.outer, label %L.exit

L.exit:
  ret void
}

; YAML-MEMCPY:      --- !Passed
; YAML-MEMCPY:      Pass:            loop-idiom
; YAML-MEMCPY:      Name:            ContiguityGuardedMemcpy
; YAML-MEMCPY:      Function:        memcpy_cont_specialized
; YAML-MEMCPY:        - NewFunction:     llvm.memcpy
; YAML-MEMCPY:        - String:          '() intrinsic'
; YAML-MEMCPY-NEXT:   - ContiguityGuarded: 'true'

; RPASS-MEMCPY: Formed contiguity-guarded memcpy
; RPASS-MEMCPY-NOT: contiguity-guarded memset

; DEBUG-MEMCPY: contiguity-guarded memcpy
; DEBUG-MEMCPY-NOT: contiguity-guarded memset

define void @memcpy_cont_specialized(ptr noalias nocapture writeonly %arr, ptr noalias nocapture readonly %ubnd) local_unnamed_addr {
L.entry:
  %0 = load i32, ptr %ubnd, align 4
  %1 = icmp slt i32 %0, 1
  %Base = alloca i32, i32 1600
  br i1 %1, label %L.exit, label %L.outer.preheader

L.outer.preheader:
  %2 = sext i32 %0 to i64
  %dst.base = getelementptr i8, ptr %arr, i64 -20
  br label %L.outer

L.outer:
  %outer.iv = phi i64 [ 100, %L.outer.preheader ], [ %outer.next, %L.outer.latch ]
  %ia = phi i64 [ 1, %L.outer.preheader ], [ %ia.next, %L.outer.latch ]
  %row.off = shl nsw i64 %ia, 2
  br label %L.inner

L.inner:
  %inner.iv = phi i64 [ %2, %L.outer ], [ %inner.dec, %L.inner ]
  %ib = phi i64 [ 1, %L.outer ], [ %ib.next, %L.inner ]
  %sum = add nuw nsw i64 %ib, %row.off
  %sp = getelementptr i32, ptr %Base, i64 %sum
  %dp = getelementptr i32, ptr %dst.base, i64 %sum
  %v = load i32, ptr %sp, align 1
  store i32 %v, ptr %dp, align 4
  %ib.next = add nuw nsw i64 %ib, 1
  %inner.dec = add nsw i64 %inner.iv, -1
  %inner.alive = icmp sgt i64 %inner.iv, 1
  br i1 %inner.alive, label %L.inner, label %L.outer.latch

L.outer.latch:
  %ia.next = add nuw nsw i64 %ia, 1
  %outer.next = add nsw i64 %outer.iv, -1
  %outer.alive = icmp sgt i64 %outer.iv, 1
  br i1 %outer.alive, label %L.outer, label %L.exit

L.exit:
  ret void
}

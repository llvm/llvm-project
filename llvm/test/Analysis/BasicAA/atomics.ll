; RUN: opt -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 %s | FileCheck %s

declare void @escape(ptr)
declare noalias ptr @malloc(i64)

; CHECK-LABEL: Function: alloca_no_escape:
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %1 = atomicrmw add ptr %x, i32 1 monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %1 = atomicrmw add ptr %x, i32 1 monotonic, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 monotonic monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 monotonic monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %3 = load atomic i32, ptr %x monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %3 = load atomic i32, ptr %x monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  store atomic i32 0, ptr %x monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %4 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %4 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %5 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %5 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %6 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %6 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x release, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  store atomic i32 0, ptr %x release, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  fence seq_cst
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  fence seq_cst
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %7 = atomicrmw add ptr %x, i32 1 seq_cst, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %7 = atomicrmw add ptr %x, i32 1 seq_cst, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %8 = cmpxchg ptr %x, i32 0, i32 1 seq_cst seq_cst, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %8 = cmpxchg ptr %x, i32 0, i32 1 seq_cst seq_cst, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %9 = load atomic i32, ptr %x seq_cst, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %9 = load atomic i32, ptr %x seq_cst, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x seq_cst, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  store atomic i32 0, ptr %x seq_cst, align 4
define void @alloca_no_escape(ptr %x) {
  %a = alloca i32
  store i32 0, ptr %a

  atomicrmw add ptr %x, i32 1 monotonic
  cmpxchg ptr %x, i32 0, i32 1 monotonic monotonic
  load atomic i32, ptr %x monotonic, align 4
  store atomic i32 0, ptr %x monotonic, align 4

  fence release
  atomicrmw add ptr %x, i32 1 acq_rel
  cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic
  load atomic i32, ptr %x acquire, align 4
  store atomic i32 0, ptr %x release, align 4

  fence seq_cst
  atomicrmw add ptr %x, i32 1 seq_cst
  cmpxchg ptr %x, i32 0, i32 1 seq_cst seq_cst
  load atomic i32, ptr %x seq_cst, align 4
  store atomic i32 0, ptr %x seq_cst, align 4

  ret void
}

; CHECK-LABEL: Function: alloca_escape_after:
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x release, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  store atomic i32 0, ptr %x release, align 4
define void @alloca_escape_after(ptr %x) {
  %a = alloca i32
  store i32 0, ptr %a

  fence release
  atomicrmw add ptr %x, i32 1 acq_rel
  cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic
  load atomic i32, ptr %x acquire, align 4
  store atomic i32 0, ptr %x release, align 4

  call void @escape(ptr %a)

  ret void
}

; CHECK-LABEL: Function: noalias_no_escape:
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x release, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  store atomic i32 0, ptr %x release, align 4
define void @noalias_no_escape(ptr noalias %a, ptr %x) {
  store i32 0, ptr %a

  fence release
  atomicrmw add ptr %x, i32 1 acq_rel
  cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic
  load atomic i32, ptr %x acquire, align 4
  store atomic i32 0, ptr %x release, align 4

  ret void
}

; CHECK-LABEL: Function: noalias_escape_after:
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x release, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  store atomic i32 0, ptr %x release, align 4
define void @noalias_escape_after(ptr noalias %a, ptr %x) {
  store i32 0, ptr %a

  fence release
  atomicrmw add ptr %x, i32 1 acq_rel
  cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic
  load atomic i32, ptr %x acquire, align 4
  store atomic i32 0, ptr %x release, align 4

  ret void
}

; CHECK-LABEL: Function: malloc_no_escape:
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %a = call ptr @malloc(i64 4)
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %a = call ptr @malloc(i64 4)
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x release, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  store atomic i32 0, ptr %x release, align 4
define void @malloc_no_escape(ptr %x) {
  %a = call ptr @malloc(i64 4)
  store i32 0, ptr %a

  fence release
  atomicrmw add ptr %x, i32 1 acq_rel
  cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic
  load atomic i32, ptr %x acquire, align 4
  store atomic i32 0, ptr %x release, align 4

  ret void
}

; CHECK-LABEL: Function: malloc_escape_after:
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %a = call ptr @malloc(i64 4)
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %a = call ptr @malloc(i64 4)
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x release, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  store atomic i32 0, ptr %x release, align 4
define ptr @malloc_escape_after(ptr %x) {
  %a = call ptr @malloc(i64 4)
  store i32 0, ptr %a

  fence release
  atomicrmw add ptr %x, i32 1 acq_rel
  cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic
  load atomic i32, ptr %x acquire, align 4
  store atomic i32 0, ptr %x release, align 4

  ret ptr %a
}

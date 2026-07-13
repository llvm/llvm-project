; RUN: opt -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 %s | FileCheck %s

declare void @escape(ptr)
declare noalias ptr @malloc(i64)
declare void @call()

; CHECK-LABEL: Function: alloca_no_escape:
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %1 = atomicrmw add ptr %x, i32 1 monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %1 = atomicrmw add ptr %x, i32 1 monotonic, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 monotonic monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 monotonic monotonic, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %3 = load atomic i32, ptr %x monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %3 = load atomic i32, ptr %x monotonic, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  store atomic i32 0, ptr %x monotonic, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  fence release
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %4 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %4 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %5 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %5 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %6 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %6 = load atomic i32, ptr %x acquire, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x release, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  store atomic i32 0, ptr %x release, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  fence seq_cst
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  fence seq_cst
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %7 = atomicrmw add ptr %x, i32 1 seq_cst, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %7 = atomicrmw add ptr %x, i32 1 seq_cst, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %8 = cmpxchg ptr %x, i32 0, i32 1 seq_cst seq_cst, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %8 = cmpxchg ptr %x, i32 0, i32 1 seq_cst seq_cst, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %9 = load atomic i32, ptr %x seq_cst, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %9 = load atomic i32, ptr %x seq_cst, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x seq_cst, align 4
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

; CHECK-LABEL: Function: alloca_escape_after_readonly:
; CHECK:  Just Ref:  Ptr: i32* %a	<->  fence acq_rel
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  fence acq_rel
; CHECK:  Just Ref:  Ptr: i32* %a	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Just Ref:  Ptr: i32* %a	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Just Ref:  Ptr: i32* %a	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Just Ref:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x release, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  store atomic i32 0, ptr %x release, align 4
define void @alloca_escape_after_readonly(ptr %x) {
  %a = alloca i32
  store i32 0, ptr %a

  fence acq_rel
  atomicrmw add ptr %x, i32 1 acq_rel
  cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic
  load atomic i32, ptr %x acquire, align 4
  store atomic i32 0, ptr %x release, align 4

  call void @escape(ptr captures(address, read_provenance) %a)

  ret void
}

; CHECK-LABEL: Function: alloca_no_escape_aliasing:
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %1 = atomicrmw add ptr %a, i32 1 monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %2 = cmpxchg ptr %a, i32 0, i32 1 monotonic monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %3 = load atomic i32, ptr %a monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %a monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %4 = atomicrmw add ptr %a, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %5 = cmpxchg ptr %a, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %6 = load atomic i32, ptr %a acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %a release, align 4
define void @alloca_no_escape_aliasing() {
  %a = alloca i32
  store i32 0, ptr %a

  atomicrmw add ptr %a, i32 1 monotonic
  cmpxchg ptr %a, i32 0, i32 1 monotonic monotonic
  load atomic i32, ptr %a monotonic, align 4
  store atomic i32 0, ptr %a monotonic, align 4

  atomicrmw add ptr %a, i32 1 acq_rel
  cmpxchg ptr %a, i32 0, i32 1 acq_rel monotonic
  load atomic i32, ptr %a acquire, align 4
  store atomic i32 0, ptr %a release, align 4

  ret void
}

; CHECK-LABEL: Function: noalias_no_escape:
; CHECK:  NoModRef:  Ptr: i32* %a	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  fence release
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x release, align 4
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

  call void @escape(ptr %a)

  ret void
}

; CHECK-LABEL: Function: malloc_no_escape:
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  %a = call ptr @malloc(i64 4)
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %a = call ptr @malloc(i64 4)
; CHECK:  NoModRef:  Ptr: i32* %a	<->  fence release
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  fence release
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %1 = atomicrmw add ptr %x, i32 1 acq_rel, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %2 = cmpxchg ptr %x, i32 0, i32 1 acq_rel monotonic, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  Both ModRef:  Ptr: i32* %x	<->  %3 = load atomic i32, ptr %x acquire, align 4
; CHECK:  NoModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr %x release, align 4
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

; CHECK-LABEL: Function: inline_asm
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  call void asm sideeffect "", "~{memory}"()
; CHECK:  NoModRef:  Ptr: i32* %a	<->  call void asm sideeffect "", "~{memory}"() #0
; CHECK:  NoModRef:  Ptr: i32* %a	<->  call void asm sideeffect "", "~{memory}"() #1
; CHECK:  NoModRef:  Ptr: i32* %a	<->  call void asm sideeffect "", "~{memory}"() #2
define ptr @inline_asm() {
  %a = call ptr @malloc(i64 4)
  store i32 0, ptr %a

  call void asm sideeffect "", "~{memory}"()
  call void asm sideeffect "", "~{memory}"() memory(read)
  call void asm sideeffect "", "~{memory}"() memory(argmem: readwrite)
  call void asm sideeffect "", "~{memory}"() nosync

  ret ptr %a
}

; CHECK-LABEL: Function: arbitrary_call
; CHECK:  NoModRef:  Ptr: i32* %a	<->  call void @call()
; CHECK:  NoModRef:  Ptr: i32* %a	<->  call void @call() #0
; CHECK:  NoModRef:  Ptr: i32* %a	<->  call void @call() #1
; CHECK:  NoModRef:  Ptr: i32* %a	<->  call void @call() #2
define ptr @arbitrary_call() {
  %a = call ptr @malloc(i64 4)
  store i32 0, ptr %a

  call void @call()
  call void @call() memory(read)
  call void @call() memory(argmem: readwrite)
  call void @call() nosync

  ret ptr %a
}

; -air-output=air writes the whole module as one wrapped AIR bitcode object,
; the .air file that Apple's metallib tool links into a library, instead of
; a .metallib container holding one module per kernel.
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o %t.air
; RUN: od -An -tx1 -N 20 %t.air | FileCheck %s --check-prefix=WRAP
; RUN: llvm-bcanalyzer -dump %t.air | FileCheck %s --check-prefix=BC
; RUN: llc -mtriple=air -filetype=obj %s -o %t.metallib
; RUN: od -An -c -N 4 %t.metallib | FileCheck %s --check-prefix=LIB

; Bitcode wrapper: magic 0x0B17C0DE, version 0, bitcode at offset 20,
; CPU type ~0 (no Darwin CPU type exists for AIR).
; WRAP: de c0 17 0b 00 00 00 00 14 00 00 00 {{.. .. .. ..}}
; WRAP-NEXT: ff ff ff ff

; Both kernels live in the single module.
; BC: <TRIPLE
; BC-COUNT-2: <FUNCTION op0=

; LIB: M T L B

declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()

define void @probe(ptr addrspace(1) %out) {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %tid = extractvalue { i32, i32, i32 } %t, 0
  %v = add i32 %tid, 7
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %v, ptr addrspace(1) %p
  ret void
}

define void @accumulate(ptr addrspace(1) %out) {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %tid = extractvalue { i32, i32, i32 } %t, 0
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  %old = load i32, ptr addrspace(1) %p
  %v = add i32 %old, 7
  store i32 %v, ptr addrspace(1) %p
  ret void
}

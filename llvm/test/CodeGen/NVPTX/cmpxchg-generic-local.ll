; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx87 -O0 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx87 -O2 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas-sm_90 && ptxas-isa-8.7 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx87 | %ptxas-verify -arch=sm_90 %}

; A generic pointer passed to a device function can designate local memory.
; Retain the native atomic for nonlocal memory and use local loads/stores for
; the thread-private case at every optimization level.
define i32 @generic_cas(ptr %p, i32 %cmp, i32 %new) noinline {
; CHECK-LABEL: generic_cas(
; CHECK: isspacep.local
; CHECK-DAG: ld.b32
; CHECK-DAG: st.b32
; CHECK-DAG: atom.relaxed.sys.cas.b32
; CHECK: ret;
  %pair = cmpxchg ptr %p, i32 %cmp, i32 %new monotonic monotonic, align 4
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

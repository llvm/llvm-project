; RUN: opt -S -passes='require<libcall-lowering-info>,expand-ir-insts,atomic-expand' %s -o - | FileCheck %s

; volatile atomicrmw shouldn't be converted to a fence
; CHECK:  %0 = atomicrmw volatile and ptr %addr, i32 -1 seq_cst

target triple = "x86_64-pc-windows-msvc"

define dso_local void @access_via_interlocked(ptr noundef %addr) {
entry:
  %0 = atomicrmw volatile and ptr %addr, i32 -1 seq_cst, align 4
  ret void
}


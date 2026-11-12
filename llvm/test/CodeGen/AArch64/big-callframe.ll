; RUN: llc -o - %s -verify-machineinstrs | FileCheck %s
; XFAIL: *
; Make sure we use a frame pointer and fp relative addressing for the emergency
; spillslot when we have gigantic callframes.
; CHECK-LABEL: func:
; CHECK: stur {{.*}}, [x29, #{{.*}}] // 8-byte Folded Spill
; CHECK: ldur {{.*}}, [x29, #{{.*}}] // 8-byte Folded Reload
target triple = "aarch64--"
declare void @extfunc(ptr byval([4096 x i64]) %p)
define void @func(ptr %z) {
  %lvar = alloca [31 x i8]
  %v = load volatile [31 x i8], ptr %lvar
  store volatile [31 x i8] %v, ptr %lvar
  call void @extfunc(ptr byval([4096 x i64]) %z)
  ret void
}

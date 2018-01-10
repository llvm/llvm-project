; RUN: llc -o - %s | FileCheck %s
; Make sure we use a frame pointer and fp relative addressing for the emergency
; spillslot when we have gigantic callframes.
; CHECK-LABEL: func:
; CHECK: stur {{.*}}, [x29, #{{.*}}] // 8-byte Folded Spill
; CHECK: ldur {{.*}}, [x29, #{{.*}}] // 8-byte Folded Reload
target triple = "aarch64--"
declare void @extfunc([4096 x i64]* byval %p)
define void @func([4096 x i64]* %z) {
  %lvar = alloca [31 x i8]
  %v = load volatile [31 x i8], [31 x i8]* %lvar
  store volatile [31 x i8] %v, [31 x i8]* %lvar
  call void @extfunc([4096 x i64]* byval %z)
  ret void
}

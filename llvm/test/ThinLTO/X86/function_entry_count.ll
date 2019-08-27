; RUN: opt -thinlto-bc %s -write-relbf-to-summary -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: opt -thinlto-bc %p/Inputs/function_entry_count.ll -write-relbf-to-summary -thin-link-bitcode-file=%t2.thinlink.bc -o %t2.bc

; First perform the thin link on the normal bitcode file.
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps -thinlto-synthesize-entry-counts \
; RUN:     -r=%t1.bc,g, \
; RUN:     -r=%t1.bc,f,px \
; RUN:     -r=%t1.bc,h,px \
; RUN:     -r=%t2.bc,h, \
; RUN:     -r=%t2.bc,g,px
; RUN: llvm-dis -o - %t.o.1.3.import.bc | FileCheck %s

; RUN: llvm-lto -thinlto-action=run -thinlto-synthesize-entry-counts -exported-symbol=f \
; RUN:     -exported-symbol=g -exported-symbol=h -thinlto-save-temps=%t3. %t1.bc %t2.bc
; RUN: llvm-dis %t3.0.3.imported.bc -o - | FileCheck %s

; CHECK: define void @h() !prof ![[PROF2:[0-9]+]]
; CHECK: define void @f(i32{{.*}}) !prof ![[PROF1:[0-9]+]]
; CHECK: define available_externally void @g() !prof ![[PROF2]]
; CHECK-DAG: ![[PROF1]] = !{!"synthetic_function_entry_count", i64 10}
; CHECK-DAG: ![[PROF2]] = !{!"synthetic_function_entry_count", i64 198}

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @g();

define void @h() {
  ret void
}

define void @f(i32 %n) {
entry:
  %cmp = icmp slt i32 %n, 1
  br i1 %cmp, label %exit, label %loop
loop:
  %n1 = phi i32 [%n, %entry], [%n2, %loop]
  call void  @g()
  %n2 = sub i32 %n1, 1
  %cmp2 = icmp slt i32 %n, 1
  br i1 %cmp2, label %exit, label %loop
exit:
  ret void
}

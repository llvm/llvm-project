; RUN: opt -mtriple=x86_64-- -early-cse -earlycse-debug-hash < %s -S | FileCheck %s

; CHECK: @foo(x86_mmx bitcast (double 0.000000e+00 to x86_mmx))

define void @bar() {
entry:
  %0 = bitcast double 0.0 to x86_mmx
  %1 = call x86_mmx @foo(x86_mmx %0)
  ret void
}

declare x86_mmx @foo(x86_mmx)

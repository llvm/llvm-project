; RUN: llc -verify-machineinstrs < %s -march=ppc32 -mtriple=powerpc-unknown-linux-gnu | \
; RUN:   grep "addic 4, 4, 1"
; RUN: llc -verify-machineinstrs < %s -march=ppc32 -mtriple=powerpc-unknown-linux-gnu | \
; RUN:   grep "addze 3, 3"

declare i64 @foo()

define i64 @bar()
{
  %t = call i64 @foo()
  %s = add i64 %t, 1
  ret i64 %s
}

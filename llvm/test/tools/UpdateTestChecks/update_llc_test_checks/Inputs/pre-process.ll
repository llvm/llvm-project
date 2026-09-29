; Test that update_llc_test_checks.py can run pre-processing commands.
; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s --check-prefix=CHECK-ADD
; RUN: sed 's/add /sub /g' %s | llc -mtriple=x86_64-unknown-unknown \
; RUN:   | FileCheck %s --check-prefix=CHECK-SUB
; Check that multiple pre-processing commands are handled
; RUN: sed 's/add /sub /g' %s | sed 's/i64 /i16 /g' | cat \
; RUN:  | llc -mtriple=x86_64-unknown-unknown | FileCheck %s --check-prefix=CHECK-SUB-I16

define i64 @test_add_constant(i64 %arg) nounwind {
entry:
  %a1 = add i64 %arg, 1
  %a2 = add i64 %a1, 2
  %a3 = add i64 %a2, 3
  ret i64 %a3
}

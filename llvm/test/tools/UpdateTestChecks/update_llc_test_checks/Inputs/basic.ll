; Example input for update_llc_test_checks (taken from CodeGen/X86/iabs.ll)
; RUN: llc < %s -mtriple=i686-unknown-unknown | FileCheck %s --check-prefix=X86 --check-prefix=X86-NO-CMOV
; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+cmov | FileCheck %s --check-prefix=X86 --check-prefix=X86-CMOV
; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s --check-prefix=X64

define i8 @test_i8(i8 %a) nounwind {
  %tmp1neg = sub i8 0, %a
  %b = icmp sgt i8 %a, -1
  %abs = select i1 %b, i8 %a, i8 %tmp1neg
  ret i8 %abs
}

define i16 @test_i16(i16 %a) nounwind {
  %tmp1neg = sub i16 0, %a
  %b = icmp sgt i16 %a, -1
  %abs = select i1 %b, i16 %a, i16 %tmp1neg
  ret i16 %abs
}

define i32 @test_i32(i32 %a) nounwind {
  %tmp1neg = sub i32 0, %a
  %b = icmp sgt i32 %a, -1
  %abs = select i1 %b, i32 %a, i32 %tmp1neg
  ret i32 %abs
}

define i64 @test_i64(i64 %a) nounwind {
  %tmp1neg = sub i64 0, %a
  %b = icmp sgt i64 %a, -1
  %abs = select i1 %b, i64 %a, i64 %tmp1neg
  ret i64 %abs
}

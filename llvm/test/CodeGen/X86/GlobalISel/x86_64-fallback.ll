; RUN: llc -O0 -mtriple=x86_64-linux-gnu -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' -verify-machineinstrs %s -o %t.out 2> %t.err
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-OUT < %t.out
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-ERR < %t.err
; This file checks that the fallback path to selection dag works.
; The test is fragile in the sense that it must be updated to expose
; something that fails with global-isel.
; When we cannot produce a test case anymore, that means we can remove
; the fallback path.

; Check that we fallback on invoke translation failures.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %vreg1<def>(s80) = G_FCONSTANT x86_fp80 0xK4002A000000000000000
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for test_x86_fp80_dump
; FALLBACK-WITH-REPORT-OUT-LABEL: test_x86_fp80_dump:
define void @test_x86_fp80_dump(x86_fp80* %ptr){
  store x86_fp80 0xK4002A000000000000000, x86_fp80* %ptr, align 16
  ret void
}


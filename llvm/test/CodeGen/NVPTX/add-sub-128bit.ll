; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefixes=COMMON,NOCARRY
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -mattr=+ptx43 | FileCheck %s --check-prefixes=COMMON,CARRY
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

; COMMON-LABEL: test_add
define i128 @test_add(i128 %a, i128 %b) {
; NOCARRY:        add.s64
; NOCARRY-NEXT:   add.s64
; NOCARRY-NEXT:   setp.lt.u64
; NOCARRY-NEXT:   selp.u64
; NOCARRY-NEXT:   add.s64

; CARRY:          add.cc.s64
; CARRY-NEXT:     addc.cc.s64

  %1 = add i128 %a, %b
  ret i128 %1
}

; COMMON-LABEL: test_sub
define i128 @test_sub(i128 %a, i128 %b) {
; NOCARRY:        sub.s64
; NOCARRY-NEXT:   setp.lt.u64
; NOCARRY-NEXT:   selp.s64
; NOCARRY-NEXT:   add.s64
; NOCARRY-NEXT:   sub.s64

; CARRY:          sub.cc.s64
; CARRY-NEXT:     subc.cc.s64

  %1 = sub i128 %a, %b
  ret i128 %1
}

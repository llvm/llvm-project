; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s 

@a = global i1 true
; no alignment

@b = global i8 1
; no alignment

@c = global i16 2
;CHECK: .p2align 1
;CHECK: c:

@d = global i32 3
;CHECK: .p2align 2
;CHECK: d:

@e = global i64 4
;CHECK: .p2align 3
;CHECK: e

@f = global float 5.0
;CHECK: .p2align 2
;CHECK: f:

@g = global double 6.0
;CHECK: .p2align 3
;CHECK: g:

@bar = common global [75 x i8] zeroinitializer, align 128
;CHECK: .comm bar,75,128

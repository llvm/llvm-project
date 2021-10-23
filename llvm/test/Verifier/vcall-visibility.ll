; RUN: not opt -S -verify < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; vcall_visibility must have either 1 or 3 operands
@vtableA = internal unnamed_addr constant { [1 x i8*] } { [1 x i8*] [i8* null]}, align 8, !vcall_visibility !{i64 2, i64 42}
; CHECK: bad !vcall_visibility attachment

; range start cannot be greater than range end
@vtableB = internal unnamed_addr constant { [1 x i8*] } { [1 x i8*] [i8* null]}, align 8, !vcall_visibility !{i64 2, i64 10, i64 8}
; CHECK: bad !vcall_visibility attachment

; vcall_visibility range cannot be over 64 bits
@vtableC = internal unnamed_addr constant { [1 x i8*] } { [1 x i8*] [i8* null]}, align 8, !vcall_visibility !{i64 2, i128 0, i128 9223372036854775808000}
; CHECK: bad !vcall_visibility attachment

; range must be two integers
@vtableD = internal unnamed_addr constant { [1 x i8*] } { [1 x i8*] [i8* null]}, align 8, !vcall_visibility !{i64 2, i64 0, !"string"}
; CHECK: bad !vcall_visibility attachment

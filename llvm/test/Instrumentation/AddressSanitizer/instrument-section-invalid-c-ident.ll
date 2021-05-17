; This test checks that sections with names not resembling valid C identifiers
; are instrumented.
; RUN: opt < %s -asan -asan-module -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-netbsd"

module asm ".hidden invalid$c$name"

@data1 = dso_local global i32 1, align 4
@data2 = dso_local global i32 2, align 4
@__invalid$c$name_sym_data1 = internal constant i8* bitcast (i32* @data1 to i8*), section "invalid$c$name", align 8
@__invalid$c$name_sym_data2 = internal constant i8* bitcast (i32* @data2 to i8*), section "invalid$c$name", align 8
; CHECK: @"__invalid$c$name_sym_data1" = internal constant{{.*}}, section "invalid$c$name"
; CHECK-NEXT: @"__invalid$c$name_sym_data2" = internal constant{{.*}}, section "invalid$c$name"
; CHECK: @"__asan_global___invalid$c$name_sym_data1"
; CHECK-NEXT: @"__asan_global___invalid$c$name_sym_data2"

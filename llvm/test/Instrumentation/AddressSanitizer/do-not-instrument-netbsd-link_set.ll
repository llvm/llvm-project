; This test checks that NetBSD link_set array elements remain consecutive.
; RUN: opt < %s -passes=asan-pipeline -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-netbsd"

module asm ".hidden __stop_link_set_test_set"

@data1 = dso_local global i32 1, align 4
@data2 = dso_local global i32 2, align 4
@__link_set_test_set_sym_data1 = internal constant i8* bitcast (i32* @data1 to i8*), section "link_set_test_set", align 8
@__link_set_test_set_sym_data2 = internal constant i8* bitcast (i32* @data2 to i8*), section "link_set_test_set", align 8
; CHECK: @__link_set_test_set_sym_data1 = internal constant i8*{{.*}}, section "link_set_test_set"
; CHECK-NEXT: @__link_set_test_set_sym_data2 = internal constant i8*{{.*}}, section "link_set_test_set"

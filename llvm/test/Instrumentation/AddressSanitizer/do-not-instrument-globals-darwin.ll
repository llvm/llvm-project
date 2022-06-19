; This test checks that we are not instrumenting unnecessary globals
; (llvm.metadata and other llvm internal globals).
; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

@foo_noinst = private global [19 x i8] c"scannerWithString:\00", section "__TEXT,__objc_methname,cstring_literals"

; CHECK: @foo_noinst = private global [19 x i8] c"scannerWithString:\00", section "__TEXT,__objc_methname,cstring_literals"

@.str_noinst = private unnamed_addr constant [4 x i8] c"aaa\00", section "llvm.metadata"
@.str_noinst_old_cov = private unnamed_addr constant [4 x i8] c"aaa\00", section "__DATA,__llvm_covmap"
@.str_noinst_new_cov = private unnamed_addr constant [4 x i8] c"aaa\00", section "__LLVM_COV,__llvm_covmap"
@.str_noinst_LLVM = private unnamed_addr constant [4 x i8] c"aaa\00", section "__LLVM,__not_visible"
@.str_inst = private unnamed_addr constant [4 x i8] c"aaa\00"

; CHECK-NOT: {{asan_gen.*str_noinst}}
; CHECK-NOT: {{asan_gen.*str_noinst_old_cov}}
; CHECK-NOT: {{asan_gen.*str_noinst_new_cov}}
; CHECK-NOT: {{asan_gen.*str_noinst_LLVM}}
; CHECK: {{asan_gen.*str_inst}}
; CHECK: @asan.module_ctor

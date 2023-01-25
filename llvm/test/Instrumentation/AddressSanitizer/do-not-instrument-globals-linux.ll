; This test checks that we are not instrumenting unnecessary globals
; (llvm.metadata, init_array sections, and other llvm internal globals).
; RUN: opt < %s -passes=asan -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define internal void @_ZL3foov() #0 {
entry:
  ret void
}

@__call_foo = global ptr @_ZL3foov, section ".preinit_array", align 8
@__call_foo_2 = global ptr @_ZL3foov, section ".init_array", align 8
@__call_foo_3 = global ptr @_ZL3foov, section ".fini_array", align 8

; CHECK-NOT: asan_gen{{.*}}__call_foo

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0
}

@.str_noinst = private unnamed_addr constant [4 x i8] c"aaa\00", section "llvm.metadata"
@.str_noinst_prof = private unnamed_addr constant [4 x i8] c"aaa\00", section "__llvm_prf_data"
@.str_inst = private unnamed_addr constant [4 x i8] c"aaa\00"

; CHECK-NOT: {{asan_gen.*str_noinst}}
; CHECK-NOT: {{asan_gen.*str_noinst_prof}}
; CHECK: {{asan_gen.*str_inst}}
; CHECK: @asan.module_ctor

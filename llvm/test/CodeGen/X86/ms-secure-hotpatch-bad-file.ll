; RUN: not llc -mtriple=x86_64-windows --ms-secure-hotpatch-functions-file=%S/this-file-is-intentionally-missing-do-not-create-it.txt < %s 2>&1 | FileCheck %s
; CHECK: failed to open hotpatch functions file

source_filename = ".\\ms-secure-hotpatch.ll"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.36.32537"

@some_global_var = external global i32

define noundef i32 @this_gets_hotpatched() #0 {
    %1 = load i32, ptr @some_global_var
    %2 = add i32 %1, 1
    ret i32 %2
}

attributes #0 = { "marked_for_windows_hot_patching" mustprogress noinline nounwind optnone uwtable }

; This tests directly annotating a function with marked_for_windows_hot_patching, but on the wrong target OS.
;
; RUN: not llc -mtriple=x86_64-unknown-linux --ms-hotpatch-functions-list=this_never_gets_used --ms-hotpatch-functions-file=this_never_gets_used < %s 2>&1 | FileCheck %s
;
; CHECK: error: --ms-hotpatch-functions-file is only supported when target OS is Windows
; CHECK: error: --ms-hotpatch-functions-list is only supported when target OS is Windows
; CHECK: error: function is marked for Windows hot-patching, but the target OS is not Windows: this_gets_hotpatched

source_filename = ".\\ms-hotpatch-bad-os.ll"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

define noundef i32 @this_gets_hotpatched() #0 {
    ret i32 0
}

attributes #0 = { marked_for_windows_hot_patching mustprogress noinline nounwind optnone uwtable }

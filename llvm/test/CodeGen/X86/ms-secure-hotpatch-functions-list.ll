; This tests annotating a function with marked_for_windows_hot_patching by using --ms-hotpatch-functions-list.
;
; RUN: llc -mtriple=x86_64-windows --ms-secure-hotpatch-functions-list=this_gets_hotpatched < %s | FileCheck %s

source_filename = ".\\ms-secure-hotpatch-functions-list.ll"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.36.32537"

@some_global_var = external global i32

define noundef i32 @this_gets_hotpatched() #0 {
    %1 = load i32, ptr @some_global_var
    %2 = add i32 %1, 1
    ret i32 %2
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable }

; CHECK: this_gets_hotpatched: # @this_gets_hotpatched
; CHECK-NEXT: bb.0:
; CHECK-NEXT: movq __ref_some_global_var(%rip), %rax
; CHECK-NEXT: movl (%rax), %eax
; CHECK-NEXT: addl $1, %eax
; CHECK-NEXT: retq

define noundef i32 @this_does_not_get_hotpatched() #1 {
    %1 = load i32, ptr @some_global_var
    %2 = add i32 %1, 1
    ret i32 %2
}

attributes #1 = { mustprogress noinline nounwind optnone uwtable }

; CHECK: this_does_not_get_hotpatched: # @this_does_not_get_hotpatched
; CHECK-NEXT: bb.0:
; CHECK-NEXT: movl some_global_var(%rip), %eax
; CHECK-NEXT: addl $1, %eax
; CHECK-NEXT: retq

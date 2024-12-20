; RUN: llc -O0 -stop-after yk-shadow-stack-pass -yk-shadow-stack < %s  | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


declare ptr @yk_mt_new();
declare ptr @yk_location_new();
%struct.YkLocation = type { i64 }

; The pass should insert a global variable to hold the shadow stack pointer.
; CHECK: @shadowstack_0 = global ptr null

; Check that a main fucntion requiring no shadow space doesn't needlessly
; fiddle with the shadow stack pointer.
;
; It should however, still allocate and initialise the shadow stack pointer.
;
; CHECK:       define dso_local i32 @main(i32 noundef %argc, ptr noundef %argv) #0 {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %0 = call ptr @malloc(i64 1000000)
; CHECK-NEXT:    store ptr %0, ptr @shadowstack_0, align 8
; CHECK-NEXT:    ret i32 0
; CHECK-NEXT:  }
define dso_local i32 @main(i32 noundef %argc, ptr noundef %argv) noinline optnone {
entry:
  ret i32 0
}

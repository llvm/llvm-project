; RUN: opt -ipconstprop -S < %s | FileCheck %s
;
;    #include <threads.h>
;    thread_local int gtl = 0;
;    int gsh = 0;
;
;    static int callee(int *thread_local_ptr, int *shared_ptr) {
;      return *thread_local_ptr + *shared_ptr;
;    }
;
;    void broker(int *, int (*callee)(int *, int *), int *);
;
;    void caller() {
;      broker(&gtl, callee, &gsh);
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@gtl = dso_local thread_local global i32 0, align 4
@gsh = dso_local global i32 0, align 4

define internal i32 @callee(i32* %thread_local_ptr, i32* %shared_ptr) {
entry:
; CHECK:  %tmp = load i32, i32* %thread_local_ptr, align 4
; CHECK:  %tmp1 = load i32, i32* @gsh, align 4
; CHECK:  %add = add nsw i32 %tmp, %tmp1
  %tmp = load i32, i32* %thread_local_ptr, align 4
  %tmp1 = load i32, i32* %shared_ptr, align 4
  %add = add nsw i32 %tmp, %tmp1
  ret i32 %add
}

define dso_local void @caller() {
entry:
  call void @broker(i32* nonnull @gtl, i32 (i32*, i32*)* nonnull @callee, i32* nonnull @gsh)
  ret void
}

declare !callback !0 dso_local void @broker(i32*, i32 (i32*, i32*)*, i32*)

!1 = !{i64 1, i64 0, i64 2, i1 false}
!0 = !{!1}

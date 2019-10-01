; RUN: opt -ipconstprop -S < %s | FileCheck %s
;
;    #include <pthread.h>
;
;    void *GlobalVPtr;
;
;    static void *foo(void *arg) { return arg; }
;    static void *bar(void *arg) { return arg; }
;
;    int main() {
;      pthread_t thread;
;      pthread_create(&thread, NULL, foo, NULL);
;      pthread_create(&thread, NULL, bar, &GlobalVPtr);
;      return 0;
;    }
;
; Verify the constant values NULL and &GlobalVPtr are propagated into foo and
; bar, respectively.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%union.pthread_attr_t = type { i64, [48 x i8] }

@GlobalVPtr = common dso_local global i8* null, align 8

define dso_local i32 @main() {
entry:
  %thread = alloca i64, align 8
  %call = call i32 @pthread_create(i64* nonnull %thread, %union.pthread_attr_t* null, i8* (i8*)* nonnull @foo, i8* null)
  %call1 = call i32 @pthread_create(i64* nonnull %thread, %union.pthread_attr_t* null, i8* (i8*)* nonnull @bar, i8* bitcast (i8** @GlobalVPtr to i8*))
  ret i32 0
}

declare !callback !0 dso_local i32 @pthread_create(i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*)

define internal i8* @foo(i8* %arg) {
entry:
; CHECK: ret i8* null
  ret i8* %arg
}

define internal i8* @bar(i8* %arg) {
entry:
; CHECK: ret i8* bitcast (i8** @GlobalVPtr to i8*)
  ret i8* %arg
}

!1 = !{i64 2, i64 3, i1 false}
!0 = !{!1}

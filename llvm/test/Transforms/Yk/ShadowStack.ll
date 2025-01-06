; Checks that the shadow stack pass does what it should.
;
; RUN: llc -O0 -stop-after yk-shadow-stack-pass -yk-shadow-stack < %s  | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


declare ptr @yk_mt_new();
declare ptr @yk_location_new();
%struct.YkLocation = type { i64 }

; The pass should insert a global variable to hold the shadow stack pointer.
; CHECK: @shadowstack_0 = global ptr null

; Check a non-main function that requires some shadow space.
;
; CHECK: define dso_local i32 @f(i32 noundef %x, i32 noundef %y, i32 noundef %z) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load ptr, ptr @shadowstack_0, align 8
; CHECK-NEXT:   %1 = getelementptr i8, ptr %0, i32 16
; CHECK-NEXT:   store ptr %1, ptr @shadowstack_0, align 8
; CHECK-NEXT:   %2 = getelementptr i8, ptr %0, i32 0
; CHECK-NEXT:   %3 = getelementptr i8, ptr %0, i32 4
; CHECK-NEXT:   %4 = getelementptr i8, ptr %0, i32 8
; CHECK-NEXT:   %5 = getelementptr i8, ptr %0, i32 12
; CHECK:       return:
; CHECK-NEXT:    %11 = load i32, ptr %2, align 4
; CHECK-NEXT:    store ptr %0, ptr @shadowstack_0, align 8
; CHECK-NEXT:    ret i32 %11
; CHECK-NEXT:  }
define dso_local i32 @f(i32 noundef %x, i32 noundef %y, i32 noundef %z) noinline optnone {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  %z.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  store i32 %y, ptr %y.addr, align 4
  store i32 %z, ptr %z.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %cmp = icmp sgt i32 %0, 3
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %1 = load i32, ptr %y.addr, align 4
  %2 = load i32, ptr %z.addr, align 4
  %add = add nsw i32 %1, %2
  store i32 %add, ptr %retval, align 4
  br label %return

if.else:
  %3 = load i32, ptr %x.addr, align 4
  %4 = load i32, ptr %y.addr, align 4
  %add1 = add nsw i32 %3, %4
  store i32 %add1, ptr %retval, align 4
  br label %return

return:
  %5 = load i32, ptr %retval, align 4
  ret i32 %5
}

; Now let's check that a function requiring no shadow space doesn't load, add 0
; to, and store back, the shadow stack pointer. To do so would be wasteful. In
; other words, the function should remain empty.
;
; CHECK:       define dso_local void @g() #0 {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
define dso_local void @g() optnone noinline {
entry:
  ret void
}

; Now a main, which has a slightly different prologue to other functions.
;
; We also check that some special values don't end up on the shadow stack.
;
; CHECK:  define dso_local i32 @main(i32 noundef %argc, ptr noundef %argv) #0 {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %0 = call ptr @malloc(i64 1000000)
; CHECK-NEXT:    %1 = getelementptr i8, ptr %0, i32 32
; CHECK-NEXT:    store ptr %1, ptr @shadowstack_0, align 8
; CHECK-NEXT:    %2 = getelementptr i8, ptr %0, i32 0
; CHECK-NEXT:    %3 = getelementptr i8, ptr %0, i32 4
; CHECK-NEXT:    %4 = getelementptr i8, ptr %0, i32 8
; CHECK-NEXT:    %5 = getelementptr i8, ptr %0, i32 16
; CHECK-NEXT:    %6 = getelementptr i8, ptr %0, i32 28
; CHECK-NEXT:    %mt_stack = alloca ptr, align 8
; CHECK-NEXT:    %loc_stack = alloca %struct.YkLocation, align 8
; CHECK:         %lrv = load i32, ptr %2, align 4
; --- remember, main() has no shadow epilogue! ---
; CHECK-NEXT:    ret i32 %lrv
; CHECK-NEXT:  }

define dso_local i32 @main(i32 noundef %argc, ptr noundef %argv) noinline optnone {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  %vs = alloca [3 x i32], align 4
  %i = alloca i32, align 4
  %mt_stack = alloca ptr, align 8 ; this should not end up on the shadow stack
  %loc_stack = alloca %struct.YkLocation, align 8 ; nor this.
  store i32 0, ptr %retval, align 4
  store i32 %argc, ptr %argc.addr, align 4
  store ptr %argv, ptr %argv.addr, align 8
  %mt = call ptr @yk_mt_new()
  store ptr %mt, ptr %mt_stack
  %loc = call ptr @yk_location_new()
  store ptr %loc, ptr %loc_stack
  %lrv = load i32, ptr %retval, align 4
  ret i32 %lrv
}

; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z1fi(i32 %i) #0 {
entry:
  %retval = alloca i32, align 4
  %i.addr = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  %0 = load i32, ptr %i.addr, align 4
  %cmp = icmp eq i32 %0, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 3, ptr %retval
  br label %return

if.end:
  %1 = load i32, ptr %i.addr, align 4
  %cmp1 = icmp eq i32 %1, 3
  br i1 %cmp1, label %if.then.2, label %if.end.3

if.then.2:
  store i32 56, ptr %retval
  br label %return

if.end.3:
  store i32 0, ptr %retval
  br label %return

return:
  %2 = load i32, ptr %retval
  ret i32 %2
}


define internal ptr @Afunc(ptr %P) {
  store i32 1, ptr %P
  store i32 3, ptr %P
  ret ptr blockaddress(@_Z1fi, %if.then.2)
}

define internal ptr @Bfunc(ptr %P) {
; CHECK-NOT: @Bfunc
  store i32 1, ptr %P
  store i32 3, ptr %P
  ret ptr blockaddress(@_Z1fi, %if.then.2)
}

; RUN: opt -S -mergefunc < %s | FileCheck %s

; We should not merge these two functions, because the blocks are different.
; This tests the handling of block addresses from different functions.
; ModuleID = '<stdin>'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


define internal i8* @Afunc(i32* %P) {
; CHECK-LABEL: @Afunc
; CHECK-NEXT: store
; CHECK-NEXT: store
; CHECK-NEXT: ret
  store i32 1, i32* %P
  store i32 3, i32* %P
  ret i8* blockaddress(@_Z1fi, %if.then)
}

define internal i8* @Bfunc(i32* %P) {
; CHECK-LABEL: @Bfunc
; CHECK-NEXT: store
; CHECK-NEXT: store
; CHECK-NEXT: ret
  store i32 1, i32* %P
  store i32 3, i32* %P
  ret i8* blockaddress(@_Z1fi, %if.then.2)
}


; Function Attrs: nounwind uwtable
define i32 @_Z1fi(i32 %i) #0 {
entry:
  %retval = alloca i32, align 4
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %cmp = icmp eq i32 %0, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 3, i32* %retval
  br label %return

if.end:
  %1 = load i32, i32* %i.addr, align 4
  %cmp1 = icmp eq i32 %1, 3
  br i1 %cmp1, label %if.then.2, label %if.end.3

if.then.2:
  store i32 56, i32* %retval
  br label %return

if.end.3:
  store i32 0, i32* %retval
  br label %return

return:
  %2 = load i32, i32* %retval
  ret i32 %2
}

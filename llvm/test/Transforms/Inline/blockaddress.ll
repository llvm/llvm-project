; RUN: opt -inline -S < %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -S < %s | FileCheck %s
; PR10162

; Make sure doit is not inlined since the blockaddress is taken
; which could be unsafe
; CHECK: store i8* blockaddress(@doit, %here), i8** %pptr, align 8

@i = global i32 1, align 4
@ptr1 = common global i8* null, align 8

define void @doit(i8** nocapture %pptr, i32 %cond) nounwind uwtable {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.end, label %here

here:
  store i8* blockaddress(@doit, %here), i8** %pptr, align 8
  br label %if.end

if.end:
  ret void
}

define void @f(i32 %cond) nounwind uwtable {
entry:
  call void @doit(i8** @ptr1, i32 %cond)
  ret void
}

; PR27233: We can inline @run into @init.  Don't crash on it.
;
; CHECK-LABEL: define void @init
; CHECK:         store i8* blockaddress(@run, %bb)
; CHECK-SAME:        @run.bb
define void @init() {
entry:
  call void @run()
  ret void
}

define void @run() {
entry:
  store i8* blockaddress(@run, %bb), i8** getelementptr inbounds ([1 x i8*], [1 x i8*]* @run.bb, i64 0, i64 0), align 8
  ret void

bb:
  unreachable
}

@run.bb = global [1 x i8*] zeroinitializer

; Check that a function referenced by a global blockaddress wont be inlined,
; even if it contains a callbr. We might be able to relax this in the future
; as long as the global blockaddress is updated correctly.
@ba = internal global i8* blockaddress(@foo, %7), align 8
define internal i32 @foo(i32) {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  callbr void asm sideeffect "testl $0, $0; jne ${1:l};", "r,!i,!i,~{dirflag},~{fpsr},~{flags}"(i32 %4) #1
          to label %5 [label %7, label %6]

; <label>:5:                                      ; preds = %1
  store i32 0, i32* %2, align 4
  br label %8

; <label>:6:                                      ; preds = %1
  store i32 1, i32* %2, align 4
  br label %8

; <label>:7:                                      ; preds = %1
  store i32 2, i32* %2, align 4
  br label %8

; <label>:8:                                      ; preds = %7, %6, %5
  %9 = load i32, i32* %2, align 4
  ret i32 %9
}
define dso_local i32 @bar() {
  %1 = call i32 @foo(i32 0)
  ret i32 %1
}

; CHECK: define dso_local i32 @bar() {
; CHECK:   %1 = call i32 @foo(i32 0)
; CHECK:   ret i32 %1
; CHECK: }

; Triple check that even with a global aggregate whose member is a blockaddress,
; we still don't inline referred to functions.

%struct.foo = type { i8* }

@my_foo = dso_local global %struct.foo { i8* blockaddress(@baz, %7) }

define internal i32 @baz(i32) {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  callbr void asm sideeffect "testl $0, $0; jne ${1:l};", "r,!i,!i,~{dirflag},~{fpsr},~{flags}"(i32 %4) #1
          to label %5 [label %7, label %6]

; <label>:5:                                      ; preds = %1
  store i32 0, i32* %2, align 4
  br label %8

; <label>:6:                                      ; preds = %1
  store i32 1, i32* %2, align 4
  br label %8

; <label>:7:                                      ; preds = %1
  store i32 2, i32* %2, align 4
  br label %8

; <label>:8:                                      ; preds = %7, %6, %5
  %9 = load i32, i32* %2, align 4
  ret i32 %9
}
define dso_local i32 @quux() {
  %1 = call i32 @baz(i32 0)
  ret i32 %1
}

; CHECK: define dso_local i32 @quux() {
; CHECK:   %1 = call i32 @baz(i32 0)
; CHECK:   ret i32 %1
; CHECK: }

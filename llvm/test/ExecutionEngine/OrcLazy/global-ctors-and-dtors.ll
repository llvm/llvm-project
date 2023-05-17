; Test that global constructors and destructors are run:
;
; RUN: lli -jit-kind=orc-lazy -orc-lazy-debug=funcs-to-stdout -extra-module %s \
; RUN:   %S/Inputs/noop-main.ll | FileCheck %s
;
; Test that this is true for global constructors and destructors in other
; JITDylibs.
; RUN: lli -jit-kind=orc-lazy -orc-lazy-debug=funcs-to-stdout \
; RUN:   -jd extra -extra-module %s -jd main %S/Inputs/noop-main.ll | FileCheck %s
;
; CHECK: Hello from constructor
; CHECK: Hello
; CHECK: [ {{.*}}main{{.*}} ]
; CHECK: Goodbye from atexit
; CHECK: Goodbye from __cxa_atexit
; CHECK: Goodbye from destructor

%class.Foo = type { i8 }

@f = global %class.Foo zeroinitializer, align 1
@__dso_handle = external global i8
@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_hello.cpp, ptr null }, { i32, ptr, ptr } { i32 1024, ptr @constructor, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @printf_wrapper, ptr null }]
@str = private unnamed_addr constant [6 x i8] c"Hello\00"
@str2 = private unnamed_addr constant [23 x i8] c"Hello from constructor\00"
@str3 = private unnamed_addr constant [24 x i8] c"Goodbye from destructor\00"
@str4 = global [26 x i8] c"Goodbye from __cxa_atexit\00"
@str5 = global [20 x i8] c"Goodbye from atexit\00"


define linkonce_odr void @_ZN3FooD1Ev(ptr nocapture readnone %this) unnamed_addr align 2 {
entry:
  %puts.i = tail call i32 @puts(ptr @str4)
  ret void
}

define void @atexit_handler() {
entry:
  %puts.i = tail call i32 @puts(ptr @str5)
  ret void
}

declare i32 @__cxa_atexit(ptr, ptr, ptr)

declare i32 @atexit(ptr)

define internal void @_GLOBAL__sub_I_hello.cpp() {
entry:
  %puts.i.i.i = tail call i32 @puts(ptr @str)
  %0 = tail call i32 @__cxa_atexit(ptr @_ZN3FooD1Ev, ptr @f, ptr @__dso_handle)
  %1 = tail call i32 @atexit(ptr @atexit_handler)
  ret void
}

define void @printf_wrapper() {
entry:
  %0 = tail call i32 @puts(ptr @str3)
  ret void
}

declare i32 @puts(ptr nocapture readonly)

define void @constructor() {
entry:
  %0 = tail call i32 @puts(ptr @str2)
  ret void
}

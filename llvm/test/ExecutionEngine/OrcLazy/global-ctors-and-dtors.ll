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
; CHECK: Goodbye
; CHECK: Goodbye again

%class.Foo = type { i8 }

@f = global %class.Foo zeroinitializer, align 1
@__dso_handle = external global i8
@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_hello.cpp, i8* null }, { i32, void ()*, i8* } { i32 1024, void ()* @constructor, i8* null }]
@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* @printf_wrapper, i8* null }]
@str = private unnamed_addr constant [6 x i8] c"Hello\00"
@str2 = private unnamed_addr constant [8 x i8] c"Goodbye\00"
@str3 = global [14 x i8] c"Goodbye again\00"
@str4 = private unnamed_addr constant [23 x i8] c"Hello from constructor\00"

define linkonce_odr void @_ZN3FooD1Ev(%class.Foo* nocapture readnone %this) unnamed_addr align 2 {
entry:
  %puts.i = tail call i32 @puts(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @str2, i64 0, i64 0))
  ret void
}

declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*)

define internal void @_GLOBAL__sub_I_hello.cpp() {
entry:
  %puts.i.i.i = tail call i32 @puts(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @str, i64 0, i64 0))
  %0 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%class.Foo*)* @_ZN3FooD1Ev to void (i8*)*), i8* getelementptr inbounds (%class.Foo, %class.Foo* @f, i64 0, i32 0), i8* @__dso_handle)
  ret void
}

define void @printf_wrapper() {
entry:
  %0 = tail call i32 @puts(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @str3, i64 0, i64 0))
  ret void
}

declare i32 @puts(i8* nocapture readonly)

define void @constructor() {
entry:
  %0 = tail call i32 @puts(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @str4, i64 0, i64 0))
  ret void
}

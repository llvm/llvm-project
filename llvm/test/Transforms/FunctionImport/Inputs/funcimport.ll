target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

@globalvar = global i32 1, align 4
@staticvar = internal global i32 1, align 4
@staticconstvar = internal unnamed_addr constant [2 x i32] [i32 10, i32 20], align 4
@commonvar = common global i32 0, align 4
@P = internal global void ()* null, align 8

@weakalias = weak alias void (...), bitcast (void ()* @globalfunc1 to void (...)*)
@analias = alias void (...), bitcast (void ()* @globalfunc2 to void (...)*)
@linkoncealias = alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)

define void @globalfunc1() #0 {
entry:
  call void @funcwithpersonality()
  call void (...) @variadic_va_start()
  ret void
}

define void @globalfunc2() #0 {
entry:
  ret void
}

define linkonce_odr void @linkoncefunc() #0 {
entry:
  ret void
}

define i32 @referencestatics(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %call = call i32 @staticfunc()
  %0 = load i32, i32* @staticvar, align 4
  %add = add nsw i32 %call, %0
  %1 = load i32, i32* %i.addr, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* @staticconstvar, i64 0, i64 %idxprom
  %2 = load i32, i32* %arrayidx, align 4
  %add1 = add nsw i32 %add, %2
  ret i32 %add1
}

define i32 @referenceglobals(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @globalfunc1()
  %0 = load i32, i32* @globalvar, align 4
  ret i32 %0
}

define i32 @referencecommon(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* @commonvar, align 4
  ret i32 %0
}

define void @setfuncptr() #0 {
entry:
  store void ()* @staticfunc2, void ()** @P, align 8
  ret void
}

define void @callfuncptr() #0 {
entry:
  %0 = load void ()*, void ()** @P, align 8
  call void %0()
  ret void
}

@weakvar = weak global i32 1, align 4
define weak void @weakfunc() #0 {
entry:
  ret void
}

define linkonce void @linkoncefunc2() #0 {
entry:
  ret void
}

define internal i32 @staticfunc() #0 {
entry:
  ret i32 1
}

declare i32 @__gxx_personality_v0(...)

; Add enough instructions to prevent import with inst limit of 5
define internal void @funcwithpersonality() #2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  ret void
}

define internal void @staticfunc2() #0 {
entry:
  ret void
}

define void @referencelargelinkonce() #0 {
entry:
  call void @linkonceodr()
  ret void
}

; A large enough linkonce_odr function that should never be imported
define linkonce_odr void @linkonceodr() #0 {
entry:
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  call void @globalfunc2()
  ret void
}

; Variadic function without va_start can be imported because inliner
; can handle it.
define void @variadic_no_va_start(...) {
    ret void
}

; Variadic function with va_start should not be imported because inliner
; doesn't handle it.
define void @variadic_va_start(...) {
    %ap = alloca i8*, align 8
    %ap.0 = bitcast i8** %ap to i8*
    call void @llvm.va_start(i8* %ap.0)
    ret void
}

declare void @llvm.va_start(i8*) nounwind

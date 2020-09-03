; RUN: opt -verify < %s 2>&1 | FileCheck %s
; CHECK-NOT: Global is marked as dllimport, but not external

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"

@"?var_hook@@3HA" = extern_weak dllimport global i32, align 4

; Function Attrs: noinline optnone uwtable
define dso_local zeroext i1 @"?foo@@YA_NPEAHH@Z"(i32* %0, i32 %1) #0 {
   ret i1 0
}

declare extern_weak dllimport void @func_hook(i32) #1

attributes #0 = { noinline optnone uwtable }
attributes #1 = { uwtable }

; Compiled from the following C++ example with --target=x86_64-pc-win32,
; using the non-checking configuration
;__declspec(dllimport) __attribute__((weak)) extern "C" void func_hook(int);
;extern __declspec(dllimport) __attribute__((weak)) int var_hook;
;bool foo(int *q, int p)
;{
;  if (func_hook)
;    func_hook(p);
;  return &var_hook == q;
;}

; This test ensures that catch parameters are not instrumented on Windows.

; This file was generated using the following source
;
; ```C++
; #include <exception>
; #include <cstdio>
;
; int main() {
;  try {
;   throw 1;
;  } catch (const int ex) {
;   printf("%d\n", ex);
;   return -1;
;  }
;  return 0;
; }
;
; ```
; then running the following sequence of commands
;
; ```
; clang.exe -g0 -O0 -emit-llvm -c main.cpp -o main.bc
; llvm-extract.exe -func=main main.bc -o main_func.bc
; llvm-dis.exe main_func.bc -o main_func_dis.ll
; ```
; and finally manually trimming the resulting `.ll` file to remove
; unnecessary metadata, and manually adding the `sanitize_address` annotation;
; needed for the ASan pass to run.

; RUN: opt < %s -passes=asan -S | FileCheck %s
; CHECK: %ex = alloca i32, align 4
; CHECK: catchpad within %{{.*}} [ptr @"??_R0H@8", i32 0, ptr %ex]

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { ptr, ptr, [3 x i8] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }

@"??_R0H@8" = external global %rtti.TypeDescriptor2
@_TI1H = external unnamed_addr constant %eh.ThrowInfo, section ".xdata"
@"??_C@_03PMGGPEJJ@?$CFd?6?$AA@" = external dso_local unnamed_addr constant [4 x i8], align 1

; Function Attrs: mustprogress noinline norecurse optnone uwtable sanitize_address
define dso_local noundef i32 @main() #0 personality ptr @__CxxFrameHandler3 {
entry:
  %retval = alloca i32, align 4
  %tmp = alloca i32, align 4
  %ex = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 1, ptr %tmp, align 4
  invoke void @_CxxThrowException(ptr %tmp, ptr @_TI1H) #2
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr @"??_R0H@8", i32 0, ptr %ex]
  %2 = load i32, ptr %ex, align 4
  %call = call i32 (ptr, ...) @printf(ptr noundef @"??_C@_03PMGGPEJJ@?$CFd?6?$AA@", i32 noundef %2) [ "funclet"(token %1) ]
  store i32 -1, ptr %retval, align 4
  catchret from %1 to label %catchret.dest

catchret.dest:                                    ; preds = %catch
  br label %return

try.cont:                                         ; No predecessors!
  store i32 0, ptr %retval, align 4
  br label %return

return:                                           ; preds = %try.cont, %catchret.dest
  %3 = load i32, ptr %retval, align 4
  ret i32 %3

unreachable:                                      ; preds = %entry
  unreachable
}

declare dso_local void @_CxxThrowException(ptr, ptr)

declare dso_local i32 @__CxxFrameHandler3(...)

; Function Attrs: mustprogress noinline optnone uwtable
declare dso_local i32 @printf(ptr noundef, ...) #1

attributes #0 = { mustprogress noinline norecurse optnone uwtable sanitize_address }
attributes #1 = { mustprogress noinline optnone uwtable }
attributes #2 = { noreturn }
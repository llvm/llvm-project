; RUN: opt < %s -passes=asan -S | FileCheck %s
; CHECK: %ex = alloca i32, align 4
; CHECK: catchpad within %{{.*}} [ptr @"??_R0H@8", i32 0, ptr %ex]

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

target triple = "x86_64-pc-windows-msvc"

@"??_R0H@8" = external global ptr

; Function Attrs: sanitize_address
define i32 @main() sanitize_address personality ptr @__CxxFrameHandler3 {
entry:
  %ex = alloca i32, align 4
  invoke void @throw()
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr @"??_R0H@8", i32 0, ptr %ex]
  call void @opaque() [ "funclet"(token %1) ]
  catchret from %1 to label %return

return:                                           ; preds = %catch
  ret i32 0

unreachable:                                      ; preds = %entry
  unreachable
}

declare void @throw() noreturn
declare void @opaque()
declare i32 @__CxxFrameHandler3(...)

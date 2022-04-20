; Test -lto-freestanding option for libLTO.
; RUN: llvm-as < %s > %t.bc

; Regular run: expects fprintf to be turned into fwrite
; RUN: llvm-lto %t.bc -exported-symbol=_foo -o %t.o
; RUN: llvm-nm %t.o | FileCheck %s --check-prefix=LTO
; LTO: fwrite

; Freestanding run: expects fprintf to NOT be turned into fwrite
; RUN: llvm-lto %t.bc -lto-freestanding -exported-symbol=_foo -o %t.o
; RUN: llvm-nm %t.o | FileCheck %s --check-prefix=LTO-FREESTANDING
; LTO-FREESTANDING: fprintf

; Test -lto-freestanding option for LTOBackend

; RUN: llvm-lto2 run -r %t.bc,_fprintf,px -r %t.bc,_hello_world,px -r %t.bc,_percent_s,px  -r %t.bc,_foo,px %t.bc -o %t1.o 2>&1
; RUN: llvm-nm %t1.o.0 | FileCheck %s --check-prefix=LTO

; RUN: llvm-lto2 run -lto-freestanding -r %t.bc,_fprintf,px -r %t.bc,_hello_world,px -r %t.bc,_percent_s,px  -r %t.bc,_foo,px %t.bc -o %t2.o 2>&1
; RUN: llvm-nm %t2.o.0 | FileCheck %s --check-prefix=LTO-FREESTANDING

; Test -lto-freestanding option for LTOBackend with custom pipeline.

; RUN: llvm-lto2 run -opt-pipeline='default<O3>' -r %t.bc,_fprintf,px -r %t.bc,_hello_world,px -r %t.bc,_percent_s,px  -r %t.bc,_foo,px %t.bc -o %t1.o 2>&1
; RUN: llvm-nm %t1.o.0 | FileCheck %s --check-prefix=LTO

; RUN: llvm-lto2 run -opt-pipeline='default<O3>' -lto-freestanding -r %t.bc,_fprintf,px -r %t.bc,_hello_world,px -r %t.bc,_percent_s,px  -r %t.bc,_foo,px %t.bc -o %t2.o 2>&1
; RUN: llvm-nm %t2.o.0 | FileCheck %s --check-prefix=LTO-FREESTANDING


target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare i32 @fprintf(%FILE*, i8*, ...)

%FILE = type { }

@hello_world = constant [13 x i8] c"hello world\0A\00"
@percent_s = constant [3 x i8] c"%s\00"

; Check fprintf(fp, "%s", str) -> fwrite(str, fp) only when builtins are enabled

define void @foo(%FILE* %fp) {
  %fmt = getelementptr [3 x i8], [3 x i8]* @percent_s, i32 0, i32 0
  %str = getelementptr [13 x i8], [13 x i8]* @hello_world, i32 0, i32 0
  call i32 (%FILE*, i8*, ...) @fprintf(%FILE* %fp, i8* %fmt, i8* %str)
  ret void
}

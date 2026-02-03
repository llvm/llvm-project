
; RUN: rm -rf %t.dir
; RUN: split-file %s %t.dir && cd %t.dir
; RUN: opt -thinlto-bc foo.ll -o foo.o
; RUN: opt -thinlto-bc mul.ll -o mul.o
; RUN: llvm-lto2 run mul.o foo.o  -o libmy.so  -r mul.o,_Z3mulii,l -r foo.o,_Z3mulii,x -r foo.o,_Z3fool,p -save-temps -filter-save-modules=mul.o
; RUN: ls | FileCheck %s --implicit-check-not=libmy.so.2 --check-prefix=MUL
; RUN: rm -f *.bc
; RUN: llvm-lto2 run mul.o foo.o  -o libmy.so  -r mul.o,_Z3mulii,l -r foo.o,_Z3mulii,x -r foo.o,_Z3fool,p -save-temps -filter-save-modules=mul.o -filter-save-modules=foo.o
; RUN: ls | FileCheck %s --check-prefix=MUL_FOO
; RUN: rm -f *.bc
; RUN: llvm-lto2 run mul.o foo.o  -o libmy.so  -r mul.o,_Z3mulii,l -r foo.o,_Z3mulii,x -r foo.o,_Z3fool,p -save-temps
; RUN: ls | FileCheck %s --check-prefix=ALL

; MUL: libmy.so.1.0.preopt.bc
; MUL: libmy.so.1.1.promote.bc
; MUL: libmy.so.1.2.internalize.bc
; MUL: libmy.so.1.3.import.bc
; MUL: libmy.so.1.4.opt.bc
; MUL: libmy.so.1.5.precodegen.bc

; MUL_FOO-COUNT-12: libmy.so.{{1|2}}.{{[0-5]}}.{{preopt|promote|internalize|import|opt|precodegen}}.bc

; ALL-COUNT-12: libmy.so.{{1|2}}.{{[0-5]}}.{{preopt|promote|internalize|import|opt|precodegen}}.bc

;--- mul.ll
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z3mulii(i32 noundef %a, i32 noundef %b) {
  %c = mul nsw i32 %a, %b
  ret i32 %c
}

;--- foo.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef i32 @_Z3fool(i32 noundef %n)  {
  %n1 = add nsw i32 %n, 1
  %mul = call noundef i32 @_Z3mulii(i32 noundef %n, i32 noundef %n1)
  ret i32 %mul
}

declare noundef i32 @_Z3mulii(i32 noundef, i32 noundef)

;; Test DTLTO output with llvm-lto2.

; RUN: rm -rf %t && split-file %s %t && cd %t

;; Generate bitcode files with summary.
; RUN: opt -thinlto-bc t1.ll -o t1.bc
; RUN: opt -thinlto-bc t2.ll -o t2.bc

;; Generate native object files.
; RUN: opt t1.ll -o t1.o
; RUN: opt t2.ll -o t2.o

;; Perform DTLTO. mock.py does not do any compilation,
;; instead it uses the native object files supplied
;; using -thinlto-distributor-arg.
; RUN: llvm-lto2 run t1.bc t2.bc -o t.o -save-temps \
; RUN:     -dtlto \
; RUN:     -dtlto-remote-opt-tool=dummy \
; RUN:     -dtlto-distributor=%python \
; RUN:     -thinlto-distributor-arg=%llvm_src_root/utils/dtlto/mock.py \
; RUN:     -thinlto-distributor-arg=t1.o \
; RUN:     -thinlto-distributor-arg=t2.o \
; RUN:     -thinlto-emit-indexes \
; RUN:     -thinlto-emit-imports \
; RUN:     -r=t1.bc,t1,px \
; RUN:     -r=t2.bc,t2,px

;; Check that the expected output files have been created.
; RUN: ls * | FileCheck %s --check-prefix=OUTPUT

; OUTPUT-DAG: t1.{{[0-9]+}}.{{[0-9]+}}.native.o{{$}}
; OUTPUT-DAG: t1.bc.imports{{$}}
; OUTPUT-DAG: t1.{{[0-9]+}}.{{[0-9]+}}.native.o.thinlto.bc{{$}}

; OUTPUT-DAG: t2.{{[0-9]+}}.{{[0-9]+}}.native.o{{$}}
; OUTPUT-DAG: t2.bc.imports{{$}}
; OUTPUT-DAG: t2.{{[0-9]+}}.{{[0-9]+}}.native.o.thinlto.bc{{$}}


;--- t1.ll

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @t1() {
  ret void
}

;--- t2.ll

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @t2() {
  ret void
}

;--- t3.ll

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown-gnu"

define void @t3() {
  ret void
}

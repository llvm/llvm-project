; Check that the JSON output from DTLTO is as expected. Note that validate.py
; checks the JSON structure so we just check the field contents in this test.

RUN: rm -rf %t && split-file %s %t && cd %t

; Generate bitcode files with summary.
RUN: opt -thinlto-bc t1.ll -o t1.bc
RUN: opt -thinlto-bc t2.ll -o t2.bc

; Perform DTLTO.
RUN: not llvm-lto2 run t1.bc t2.bc -o my.output \
RUN:     -r=t1.bc,t1,px -r=t2.bc,t2,px \
RUN:     -dtlto-distributor=%python \
RUN:     -dtlto-distributor-arg=%llvm_src_root/utils/dtlto/validate.py,--da1=10,--da2=10 \
RUN:     -dtlto-compiler=my_clang.exe \
RUN:     -dtlto-compiler-arg=--rota1=10,--rota2=20 \
RUN:   2>&1 | FileCheck %s

CHECK: distributor_args=['--da1=10', '--da2=10']

; Check the common object.
CHECK:      "linker_output": "my.output"
CHECK:      "args":
CHECK-NEXT: "my_clang.exe"
CHECK-NEXT: "-c"
CHECK-NEXT: "--target=x86_64-unknown-linux-gnu"
CHECK-NEXT: "-O2"
CHECK-NEXT: "-fpic"
CHECK-NEXT: "-Wno-unused-command-line-argument"
CHECK-NEXT: "--rota1=10"
CHECK-NEXT: "--rota2=20"
CHECK-NEXT: ]
CHECK: "inputs": []

; Check the first job entry.
CHECK:      "args":
CHECK-NEXT: "t1.bc"
CHECK-NEXT: "-fthinlto-index=t1.1.[[#]].native.o.thinlto.bc"
CHECK-NEXT: "-o"
CHECK-NEXT: "t1.1.[[#]].native.o"
CHECK-NEXT: ]
CHECK:      "inputs": [
CHECK-NEXT: "t1.bc"
CHECK-NEXT: "t1.1.[[#]].native.o.thinlto.bc"
CHECK-NEXT: ]
CHECK:      "outputs": [
CHECK-NEXT: "t1.1.[[#]].native.o"
CHECK-NEXT: ]

; Check the second job entry.
CHECK:      "args": [
CHECK-NEXT: "t2.bc"
CHECK-NEXT: "-fthinlto-index=t2.2.[[#]].native.o.thinlto.bc"
CHECK-NEXT: "-o"
CHECK-NEXT: "t2.2.[[#]].native.o"
CHECK-NEXT: ]
CHECK-NEXT: "inputs": [
CHECK-NEXT: "t2.bc"
CHECK-NEXT: "t2.2.[[#]].native.o.thinlto.bc"
CHECK-NEXT: ]
CHECK-NEXT: "outputs": [
CHECK-NEXT: "t2.2.[[#]].native.o"
CHECK-NEXT: ]

; This check ensures that we have failed for the expected reason.
CHECK: failed: DTLTO backend compilation: cannot open native object file:

;--- t1.ll
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @t1() {
entry:
  ret void
}

;--- t2.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @t2() {
entry:
  ret void
}

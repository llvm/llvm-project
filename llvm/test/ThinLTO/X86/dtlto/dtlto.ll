; Test DTLTO output with llvm-lto2.

RUN: rm -rf %t && split-file %s %t && cd %t

; Generate bitcode files with summary.
RUN: opt -thinlto-bc t1.ll -o t1.bc
RUN: opt -thinlto-bc t2.ll -o t2.bc

; Generate fake object files for mock.py to return.
RUN: touch t1.o t2.o

; Create an empty subdirectory to avoid having to account for the input files.
RUN: mkdir %t/out && cd %t/out

; Define a substitution to share the common DTLTO arguments.
DEFINE: %{command} = llvm-lto2 run ../t1.bc ../t2.bc -o t.o \
DEFINE:   -dtlto-distributor=%python \
DEFINE:   -dtlto-distributor-arg=%llvm_src_root/utils/dtlto/mock.py,../t1.o,../t2.o \
DEFINE:   -r=../t1.bc,t1,px \
DEFINE:   -r=../t2.bc,t2,px

; Perform DTLTO. mock.py does not do any compilation, instead it simply writes
; the contents of the object files supplied on the command line into the
; output object files in job order.
RUN: %{command}

; Check that the expected output files have been created.
RUN: ls | count 2
RUN: ls | FileCheck %s --check-prefix=THINLTO

; llvm-lto2 ThinLTO output files.
THINLTO-DAG: {{^}}t.o.1{{$}}
THINLTO-DAG: {{^}}t.o.2{{$}}

RUN: cd .. && rm -rf %t/out && mkdir %t/out && cd %t/out

; Perform DTLTO with --save-temps.
RUN: %{command} --save-temps

; Check that the expected output files have been created.
RUN: ls | count 12
RUN: ls | FileCheck %s --check-prefixes=THINLTO,SAVETEMPS

; Common -save-temps files from llvm-lto2.
SAVETEMPS-DAG: {{^}}t.o.resolution.txt{{$}}
SAVETEMPS-DAG: {{^}}t.o.index.bc{{$}}
SAVETEMPS-DAG: {{^}}t.o.index.dot{{$}}

; -save-temps incremental files.
SAVETEMPS-DAG: {{^}}t.o.0.0.preopt.bc{{$}}
SAVETEMPS-DAG: {{^}}t.o.0.2.internalize.bc{{$}}

; A jobs description JSON.
SAVETEMPS-DAG: {{^}}t.[[#]].dist-file.json{{$}}

; Summary shards emitted for DTLTO.
SAVETEMPS-DAG: {{^}}t1.1.[[#]].native.o.thinlto.bc{{$}}
SAVETEMPS-DAG: {{^}}t2.2.[[#]].native.o.thinlto.bc{{$}}

; DTLTO native output files (the results of the external backend compilations).
SAVETEMPS-DAG: {{^}}t1.1.[[#]].native.o{{$}}
SAVETEMPS-DAG: {{^}}t2.2.[[#]].native.o{{$}}

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

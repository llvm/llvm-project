; Test DTLTO output with llvm-lto2.

RUN: rm -rf %t && split-file %s %t && cd %t

; Generate bitcode files with summary.
RUN: opt -thinlto-bc t1.ll -o t1.bc
RUN: opt -thinlto-bc t2.ll -o t2.bc

; Generate fake object files for mock.py to return.
RUN: touch t1.o t2.o

; Create an empty subdirectory to avoid having to account for the input files.
RUN: mkdir %t/out && cd %t/out

; Define a substitution to share the common DTLTO arguments with caching enabled.
DEFINE: %{command} = llvm-lto2 run ../t1.bc ../t2.bc -o t.o -cache-dir cache-dir \
DEFINE:   -dtlto-distributor=%python \
DEFINE:   -dtlto-distributor-arg=%llvm_src_root/utils/dtlto/mock.py,../t1.o,../t2.o \
DEFINE:   -r=../t1.bc,t1,px \
DEFINE:   -r=../t2.bc,t2,px

; Perform out of process ThinLTO (DTLTO). 
; Note: mock.py does not do any compilation, instead it simply writes
; the contents of the object files supplied on the command line into the
; output object files in job order.
RUN: %{command}

; Check that the expected output files have been created.
RUN: ls | count 3
; Check that two native object files has been created
RUN: ls | FileCheck %s --check-prefix=THINLTO
; Check that DTLTO cache directory has been created
RUN: ls cache-dir/* | count 2
; Check that 2 cache entries are created
RUN: ls cache-dir/llvmcache-* | count 2

; llvm-lto2 ThinLTO output files.
THINLTO-DAG: {{^}}t.o.1{{$}}
THINLTO-DAG: {{^}}t.o.2{{$}}

# Execute llvm-lto2 again and check that a fully populated cache is used correctly, 
# i.e., no additional cache entries are created for cache hits.

RUN: %{command}

RUN: ls | count 3
RUN: ls | FileCheck %s --check-prefix=THINLTO
RUN: ls cache-dir/* | count 2
RUN: ls cache-dir/llvmcache-* | count 2

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

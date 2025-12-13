; Check that DTLTO creates identical summary index shard files as are created
; for an equivalent ThinLTO link.

RUN: rm -rf %t && split-file %s %t && cd %t

; Generate ThinLTO bitcode files.
RUN: opt -thinlto-bc t1.ll -o t1.bc
RUN: opt -thinlto-bc t2.ll -o t2.bc

; Generate fake object files for mock.py to return.
RUN: touch t1.o t2.o

; Define a substitution to share the common arguments.
DEFINE: %{command} = llvm-lto2 run t1.bc t2.bc -o t.o \
DEFINE:     -r=t1.bc,t1,px \
DEFINE:     -r=t2.bc,t2,px \
DEFINE:     -r=t2.bc,t1 \
DEFINE:     -thinlto-emit-indexes

; Perform DTLTO.
RUN: %{command} \
RUN:     -dtlto-distributor=%python \
RUN:     -dtlto-distributor-arg=%llvm_src_root/utils/dtlto/mock.py,t1.o,t2.o

; Perform ThinLTO.
RUN: %{command}

; Check for equivalence. We use a wildcard to account for the PID.
RUN: cmp t1.1.*.native.o.thinlto.bc t1.bc.thinlto.bc
RUN: cmp t2.2.*.native.o.thinlto.bc t2.bc.thinlto.bc

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

declare void @t1(...)

define void @t2() {
entry:
  call void (...) @t1()
  ret void
}

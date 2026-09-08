; Check that DTLTO creates equivalent summary index shards to an ordinary
; ThinLTO link and embeds the serialized LTO configuration as metadata.

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

; Check the underlying indexes for equivalence. We use a wildcard to account
; for the PID in the DTLTO filenames.
RUN: llvm-dis t1.1.*.native.o.thinlto.bc -o - | grep '^\^' > t1.dtlto.ll
RUN: llvm-dis t1.bc.thinlto.bc -o - | grep '^\^' > t1.thinlto.ll
RUN: cmp t1.dtlto.ll t1.thinlto.ll
RUN: llvm-dis t2.2.*.native.o.thinlto.bc -o - | grep '^\^' > t2.dtlto.ll
RUN: llvm-dis t2.bc.thinlto.bc -o - | grep '^\^' > t2.thinlto.ll
RUN: cmp t2.dtlto.ll t2.thinlto.ll

; Check that each DTLTO index contains the configuration metadata.
RUN: llvm-bcanalyzer -dump t1.1.*.native.o.thinlto.bc | FileCheck %s --check-prefix=CONFIG
RUN: llvm-bcanalyzer -dump t2.2.*.native.o.thinlto.bc | FileCheck %s --check-prefix=CONFIG

; CONFIG: <METADATA_BLOCK
; CONFIG: record string = 'llvm.lto.config'
; CONFIG: </METADATA_BLOCK>

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

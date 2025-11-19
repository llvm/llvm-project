; Check that DTLTO creates imports lists correctly.

RUN: rm -rf %t && split-file %s %t && cd %t

; Generate ThinLTO bitcode files.
RUN: opt -thinlto-bc 0.ll -o 0.bc -O2
RUN: opt -thinlto-bc 1.ll -o 1.bc -O2

; Define a substitution to share the common DTLTO arguments. Note that the use
; of validate.py will cause a failure as it does not create output files.
DEFINE: %{command} = llvm-lto2 run 0.bc 1.bc -o t.o \
DEFINE:    -dtlto-distributor=%python \
DEFINE:    -dtlto-distributor-arg=%llvm_src_root/utils/dtlto/validate.py \
DEFINE:    -r=0.bc,g,px \
DEFINE:    -r=1.bc,f,px \
DEFINE:    -r=1.bc,g

; We expect an import from 0.bc into 1.bc but no imports into 0.bc. Check that
; the expected input files have been added to the JSON to account for this.
RUN: not %{command} 2>&1 | FileCheck %s --check-prefixes=INPUTS,ERR

; 1.bc should not appear in the list of inputs for 0.bc.
INPUTS:      "jobs":
INPUTS:      "inputs": [
INPUTS-NEXT:   "0.bc",
INPUTS-NEXT:   "0.1.[[#]].native.o.thinlto.bc"
INPUTS-NEXT: ]

; 0.bc should appear in the list of inputs for 1.bc.
INPUTS:      "inputs": [
INPUTS-NEXT:   "1.bc",
INPUTS-NEXT:   "1.2.[[#]].native.o.thinlto.bc",
INPUTS-NEXT:   "0.bc"
INPUTS-NEXT: ]

; This check ensures that we have failed for the expected reason.
ERR: failed: DTLTO backend compilation: cannot open native object file:


; Check that imports files are not created even if -save-temps is active.
RUN: not %{command} -save-temps 2>&1 \
RUN:   | FileCheck %s --check-prefixes=ERR
RUN: ls | FileCheck %s --check-prefix=NOIMPORTSFILES
NOIMPORTSFILES-NOT: imports


; Check that imports files are created with -thinlto-emit-imports.
RUN: not %{command} -thinlto-emit-imports 2>&1 \
RUN:   | FileCheck %s --check-prefixes=ERR
RUN: ls | FileCheck %s --check-prefix=IMPORTSFILES
IMPORTSFILES: 0.bc.imports
IMPORTSFILES: 1.bc.imports

;--- 0.ll
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @g() {
entry:
  ret void
}

;--- 1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}

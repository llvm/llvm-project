; Test that DTLTO uses the target triple from the first file in the link.

RUN: rm -rf %t && split-file %s %t && cd %t

; Generate bitcode files with summary.
RUN: opt -thinlto-bc t1.ll -o t1.bc
RUN: opt -thinlto-bc t2.ll -o t2.bc

; Define a substitution to share the common DTLTO arguments. Note that the use
; of validate.py will cause a failure as it does not create output files.
DEFINE: %{command} = llvm-lto2 run -o t.o -save-temps \
DEFINE:    -dtlto-distributor=%python \
DEFINE:    -dtlto-distributor-arg=%llvm_src_root/utils/dtlto/validate.py \
DEFINE:    -r=t1.bc,t1,px \
DEFINE:    -r=t2.bc,t2,px

; Test case where t1.bc is first.
RUN: not %{command} t1.bc t2.bc 2>&1 | FileCheck %s \
RUN:   --check-prefixes=TRIPLE1,ERR --implicit-check-not=--target
TRIPLE1: --target=x86_64-unknown-linux-gnu

; Test case where t2.bc is first.
RUN: not %{command} t2.bc t1.bc 2>&1 | FileCheck %s \
RUN:   --check-prefixes=TRIPLE2,ERR --implicit-check-not=--target
TRIPLE2: --target=x86_64-unknown-unknown-gnu

; This check ensures that we have failed for the expected reason.
ERR: failed: DTLTO backend compilation: cannot open native object file:

;--- t1.ll

target triple = "x86_64-unknown-linux-gnu"

define void @t1() {
  ret void
}

;--- t2.ll

target triple = "x86_64-unknown-unknown-gnu"

define void @t2() {
  ret void
}

; Check that we can parse assembly.
; REQUIRES: x86-registered-target

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-reduce --test=llvm-dis %t.bc

target triple = "x86_64-grtev4-linux-gnu"

module asm
    ""
define void @foo(ptr , ptr ) {
  ret void
}
^0 = module: (path: "/tmp/test.bc", hash: (1638052558, 3438078092, 2356962112, 743220300, 68728585))
^1 = gv: (name: "foo", summaries: (function: (module: ^0, flags: (linkage: linkonce_odr), insts: 1)))

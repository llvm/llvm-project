; Test to ensure that we always select the same copy of a linkonce function
; when it is encountered with different thresholds. When we encounter the
; copy in funcimport_resolved1.ll with a higher threshold via the direct call
; from main(), it will be selected for importing. When we encounter it with a
; lower threshold by reaching it from the deeper call chain via foo(), it
; won't be selected for importing. We don't want to select both the copy from
; funcimport_resolved1.ll and the smaller one from funcimport_resolved2.ll,
; leaving it up to the backend to figure out which one to actually import.
; The linkonce_odr may have different instruction counts in practice due to
; different inlines in the compile step.

; Require asserts so we can use -debug-only
; REQUIRES: asserts

; REQUIRES: x86-registered-target

; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport_resolved1.ll -o %t2.bc
; RUN: opt -module-summary %p/Inputs/funcimport_resolved2.ll -o %t3.bc

; First verify that all callees are imported with the default instruction limit
; RUN: llvm-lto2 run %t.bc %t2.bc %t3.bc -o %t4 -r=%t.bc,_main,pl -r=%t.bc,_linkonceodrfunc,l -r=%t.bc,_foo,l -r=%t2.bc,_foo,pl -r=%t2.bc,_linkonceodrfunc,pl -r=%t2.bc,_linkonceodrfunc2,pl -r=%t3.bc,_linkonceodrfunc,l -thinlto-threads=1 -debug-only=function-import 2>&1 | FileCheck %s --check-prefix=INSTLIMDEFAULT
; INSTLIMDEFAULT: Is importing function {{.*}} foo from {{.*}}funcimport_resolved1.ll
; INSTLIMDEFAULT: Is importing function {{.*}} linkonceodrfunc from {{.*}}funcimport_resolved1.ll
; INSTLIMDEFAULT: Is importing function {{.*}} linkonceodrfunc2 from {{.*}}funcimport_resolved1.ll
; INSTLIMDEFAULT: Is importing function {{.*}} f from {{.*}}funcimport_resolved1.ll
; INSTLIMDEFAULT-NOT: Is importing function {{.*}} linkonceodrfunc from {{.*}}funcimport_resolved2.ll

; Now run with the lower threshold that will only allow linkonceodrfunc to be
; imported from funcimport_resolved1.ll when encountered via the direct call
; from main(). Ensure we don't also select the copy in funcimport_resolved2.ll
; when it is encountered via the deeper call chain.
; RUN: llvm-lto2 run %t.bc %t2.bc %t3.bc -o %t4 -r=%t.bc,_main,pl -r=%t.bc,_linkonceodrfunc,l -r=%t.bc,_foo,l -r=%t2.bc,_foo,pl -r=%t2.bc,_linkonceodrfunc,pl -r=%t2.bc,_linkonceodrfunc2,pl -r=%t3.bc,_linkonceodrfunc,l -thinlto-threads=1 -debug-only=function-import -import-instr-limit=8 2>&1 | FileCheck %s --check-prefix=INSTLIM8
; INSTLIM8: Is importing function {{.*}} foo from {{.*}}funcimport_resolved1.ll
; INSTLIM8: Is importing function {{.*}} linkonceodrfunc from {{.*}}funcimport_resolved1.ll
; INSTLIM8: Not importing function {{.*}} linkonceodrfunc2 from {{.*}}funcimport_resolved1.ll
; INSTLIM8: Is importing function {{.*}} f from {{.*}}funcimport_resolved1.ll
; INSTLIM8-NOT: Is importing function {{.*}} linkonceodrfunc from {{.*}}funcimport_resolved2.ll

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define i32 @main() #0 {
entry:
  call void (...) @foo()
  call void (...) @linkonceodrfunc()
  ret i32 0
}

declare void @foo(...) #1
declare void @linkonceodrfunc(...) #1

; Test to ensure that we always select the same copy of a linkonce function
; when it is encountered with different thresholds. Initially, we will encounter
; the copy in funcimport_resolved1.ll with a lower threshold by reaching it
; from the deeper call chain via foo(), and it won't be selected for importing.
; Later we encounter it with a higher threshold via the direct call from main()
; and it will be selected. We don't want to select both the copy from
; funcimport_resolved1.ll and the smaller one from funcimport_resolved2.ll,
; leaving it up to the backend to figure out which one to actually import.
; The linkonce_odr may have different instruction counts in practice due to
; different inlines in the compile step.

; Require asserts so we can use -debug-only
; REQUIRES: asserts

; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport_resolved1.ll -o %t2.bc
; RUN: opt -module-summary %p/Inputs/funcimport_resolved2.ll -o %t3.bc

; First do a sanity check that all callees are imported with the default
; instruction limit
; RUN: llvm-lto2 run %t.bc %t2.bc %t3.bc -o %t4 -r=%t.bc,_main,pl -r=%t.bc,_linkonceodrfunc,l -r=%t.bc,_foo,l -r=%t2.bc,_foo,pl -r=%t2.bc,_linkonceodrfunc,pl -r=%t2.bc,_linkonceodrfunc2,pl -r=%t3.bc,_linkonceodrfunc,l -debug-only=function-import 2>&1 | FileCheck %s --check-prefix=INSTLIMDEFAULT
; INSTLIMDEFAULT-DAG: Is importing function {{.*}} foo from {{.*}}funcimport_resolved1.ll
; INSTLIMDEFAULT-DAG: Is importing function {{.*}} linkonceodrfunc from {{.*}}funcimport_resolved1.ll
; INSTLIMDEFAULT-DAG: Is importing function {{.*}} linkonceodrfunc2 from {{.*}}funcimport_resolved1.ll
; INSTLIMDEFAULT-DAG: Is importing function {{.*}} f from {{.*}}funcimport_resolved1.ll

; Now run with the lower threshold that will only allow linkonceodrfunc to be
; imported from funcimport_resolved1.ll the second time it is encountered.
; RUN: llvm-lto2 run %t.bc %t2.bc %t3.bc -o %t4 -r=%t.bc,_main,pl -r=%t.bc,_linkonceodrfunc,l -r=%t.bc,_foo,l -r=%t2.bc,_foo,pl -r=%t2.bc,_linkonceodrfunc,pl -r=%t2.bc,_linkonceodrfunc2,pl -r=%t3.bc,_linkonceodrfunc,l -debug-only=function-import -import-instr-limit=8 2>&1 | FileCheck %s --check-prefix=INSTLIM8
; INSTLIM8-DAG: Is importing function {{.*}} foo from {{.*}}funcimport_resolved1.ll
; INSTLIM8-DAG: Is importing function {{.*}} linkonceodrfunc from {{.*}}funcimport_resolved1.ll
; INSTLIM8-DAG: Not importing function {{.*}} linkonceodrfunc2 from {{.*}}funcimport_resolved1.ll
; INSTLIM8-DAG: Is importing function {{.*}} f from {{.*}}funcimport_resolved1.ll

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define i32 @main() #0 {
entry:
  call void (...) @foo()
  call void (...) @linkonceodrfunc()
  ret i32 0
}

declare void @foo(...) #1
declare void @linkonceodrfunc(...) #1

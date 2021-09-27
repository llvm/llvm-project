; This test ensures that we drop the preempted copy of @f from %t2.bc from its
; comdat after making it available_externally. If not we would get a
; verification error.
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/linkonce_resolution_comdat.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=run -disable-thinlto-funcattrs=0 %t1.bc %t2.bc -exported-symbol=f -exported-symbol=g -thinlto-save-temps=%t3.

; RUN: llvm-dis %t3.0.3.imported.bc -o - | FileCheck %s --check-prefix=IMPORT1
; RUN: llvm-dis %t3.1.3.imported.bc -o - | FileCheck %s --check-prefix=IMPORT2
; Copy from first module is prevailing and converted to weak_odr, copy
; from second module is preempted and converted to available_externally and
; removed from comdat.
; IMPORT1: define weak_odr i32 @f(i8* %0) unnamed_addr [[ATTR:#[0-9]+]] comdat($c1) {
; IMPORT2: define available_externally i32 @f(i8* %0) unnamed_addr [[ATTR:#[0-9]+]] {

; CHECK-DAG: attributes [[ATTR]] = { norecurse nounwind }

; RUN: llvm-nm -o - < %t1.bc.thinlto.o | FileCheck %s --check-prefix=NM1
; NM1: W f

; RUN: llvm-nm -o - < %t2.bc.thinlto.o | FileCheck %s --check-prefix=NM2
; f() would have been turned into available_externally since it is preempted,
; and inlined into g()
; NM2-NOT: f

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$c1 = comdat any

define linkonce_odr i32 @f(i8*) unnamed_addr comdat($c1) {
    ret i32 43
}

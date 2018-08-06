; Require asserts for -debug-only
; REQUIRES: asserts

; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/deadstrip.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc

; RUN: llvm-lto -exported-symbol=_main -thinlto-action=promote %t1.bc -thinlto-index=%t.index.bc -o - | llvm-lto -exported-symbol=_main -thinlto-action=internalize -thinlto-index %t.index.bc -thinlto-module-id=%t1.bc - -o - | llvm-dis -o - | FileCheck %s
; RUN: llvm-lto -exported-symbol=_main -thinlto-action=promote %t2.bc -thinlto-index=%t.index.bc -o - | llvm-lto -exported-symbol=_main -thinlto-action=internalize -thinlto-index %t.index.bc -thinlto-module-id=%t2.bc - -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK2

; RUN: llvm-lto -exported-symbol=_main -thinlto-action=run %t1.bc %t2.bc
; RUN: llvm-nm %t1.bc.thinlto.o | FileCheck %s --check-prefix=CHECK-NM

; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.out -save-temps \
; RUN:   -r %t1.bc,_main,plx \
; RUN:   -r %t1.bc,_bar,pl \
; RUN:   -r %t1.bc,_dead_func,pl \
; RUN:   -r %t1.bc,_baz,l \
; RUN:   -r %t1.bc,_boo,l \
; RUN:   -r %t1.bc,_live_available_externally_func,l \
; RUN:   -r %t2.bc,_baz,pl \
; RUN:   -r %t2.bc,_boo,pl \
; RUN:   -r %t2.bc,_dead_func,l \
; RUN:   -r %t2.bc,_another_dead_func,pl \
; RUN:   -thinlto-threads=1 \
; RUN:	 -debug-only=function-import 2>&1 | FileCheck %s --check-prefix=DEBUG
; RUN: llvm-dis < %t.out.1.3.import.bc | FileCheck %s --check-prefix=LTO2
; RUN: llvm-dis < %t.out.2.3.import.bc | FileCheck %s --check-prefix=LTO2-CHECK2
; RUN: llvm-nm %t.out.1 | FileCheck %s --check-prefix=CHECK2-NM

; RUN: llvm-bcanalyzer -dump %t.out.index.bc | FileCheck %s --check-prefix=COMBINED
; Live, NotEligibleForImport, dso_local, Internal
; COMBINED-DAG: <COMBINED {{.*}} op2=119
; Live, dso_local, Internal
; COMBINED-DAG: <COMBINED {{.*}} op2=103
; Live, Local, AvailableExternally
; COMBINED-DAG: <COMBINED {{.*}} op2=97
; Live, Local, External
; COMBINED-DAG: <COMBINED {{.*}} op2=96
; COMBINED-DAG: <COMBINED {{.*}} op2=96
; COMBINED-DAG: <COMBINED {{.*}} op2=96
; Local, (Dead)
; COMBINED-DAG: <COMBINED {{.*}} op2=64
; COMBINED-DAG: <COMBINED {{.*}} op2=64
; COMBINED-DAG: <COMBINED {{.*}} op2=64

; Dead-stripping on the index allows to internalize these,
; and limit the import of @baz thanks to early pruning.
; CHECK-NOT: available_externally {{.*}} @baz()
; CHECK: @llvm.global_ctors =
; CHECK: define internal void @_GLOBAL__I_a()
; CHECK: define internal void @bar() {
; CHECK: define internal void @bar_internal()
; CHECK: define internal void @dead_func() {
; CHECK-NOT: available_externally {{.*}} @baz()
; LTO2-NOT: available_externally {{.*}} @baz()
; LTO2: @llvm.global_ctors =
; LTO2: define internal void @_GLOBAL__I_a()
; LTO2: define internal void @bar() {
; LTO2: define internal void @bar_internal()
; LTO2-NOT: @dead_func()
; LTO2-NOT: available_externally {{.*}} @baz()

; Make sure we didn't internalize @boo, which is reachable via
; llvm.global_ctors
; CHECK2: define void @boo()
; LTO2-CHECK2: define dso_local void @boo()
; We should have eventually removed @baz since it was internalized and unused
; CHECK2-NM-NOT: _baz

; The final binary should not contain any of the dead functions,
; only main is expected because bar is expected to be inlined and stripped out.
; CHECK-NM-NOT: bar
; CHECK-NM-NOT: dead
; CHECK-NM: T _main
; CHECK-NM-NOT: bar
; CHECK-NM-NOT: dead

; DEBUG-DAG: Live root: 2412314959268824392 (llvm.global_ctors)
; DEBUG-DAG: Live root: 15822663052811949562 (main)
; DEBUG-DAG: Ignores Dead GUID: 7342339837106705152 (dead_func)
; DEBUG-DAG: Ignores Dead GUID: 7546896869197086323 (baz)
; DEBUG-DAG: Initialize import for 15611644523426561710 (boo)
; DEBUG-DAG: Ignores Dead GUID: 2384416018110111308 (another_dead_func)

; Next test the case where Inputs/deadstrip.ll does not get a module index,
; which will cause it to be handled by regular LTO in the new LTO API.
; In that case there are uses of @dead_func in the regular LTO partition
; and it shouldn't be internalized.
; RUN: opt %p/Inputs/deadstrip.ll -o %t3.bc
; RUN: llvm-lto2 run %t1.bc %t3.bc -o %t4.out -save-temps \
; RUN:   -r %t1.bc,_main,plx \
; RUN:   -r %t1.bc,_bar,pl \
; RUN:   -r %t1.bc,_dead_func,pl \
; RUN:   -r %t1.bc,_baz,l \
; RUN:   -r %t1.bc,_boo,l \
; RUN:   -r %t1.bc,_live_available_externally_func,l \
; RUN:   -r %t3.bc,_baz,pl \
; RUN:   -r %t3.bc,_boo,pl \
; RUN:   -r %t3.bc,_dead_func,l \
; RUN:   -r %t3.bc,_another_dead_func,pl
; RUN: llvm-dis < %t4.out.1.3.import.bc | FileCheck %s --check-prefix=CHECK-NOTDEAD
; RUN: llvm-nm %t4.out.0 | FileCheck %s --check-prefix=CHECK-NM-NOTDEAD

; We can't internalize @dead_func because of the use in the regular LTO
; partition.
; CHECK-NOTDEAD: define dso_local void @dead_func()
; We also can't eliminate @baz because it is in the regular LTO partition
; and called from @dead_func.
; CHECK-NM-NOTDEAD: T _baz

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"


@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

declare void @baz()

declare void @boo()

define internal void @_GLOBAL__I_a() #1 section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
    call void @boo()
    ret void
}

define void @bar() {
    ret void
}

define internal void @bar_internal() {
    ret void
}

define void @dead_func() {
    call void @bar()
    call void @baz()
    call void @bar_internal()
    ret void
}

define available_externally void @live_available_externally_func() {
    ret void
}

define void @main() {
    call void @bar()
    call void @bar_internal()
    call void @live_available_externally_func()
    ret void
}

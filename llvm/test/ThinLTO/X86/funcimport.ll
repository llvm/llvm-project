; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc

; RUN: llvm-lto -thinlto-index-stats %t3.bc | FileCheck %s -check-prefix=STATS
; STATS: Index {{.*}} contains 24 nodes (13 functions, 3 alias, 8 globals) and 19 edges (8 refs and 11 calls)

; Ensure statics are promoted/renamed correctly from this file (all but
; constant variable need promotion).
; RUN: llvm-lto -thinlto-action=promote %t.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=EXPORTSTATIC
; EXPORTSTATIC-DAG: @staticvar.llvm.0 = hidden global
; Eventually @staticconstvar can be exported as a copy and not promoted
; EXPORTSTATIC-DAG: @staticconstvar.llvm.0 = hidden unnamed_addr constant
; EXPORTSTATIC-DAG: @P.llvm.0 = hidden global void ()* null
; EXPORTSTATIC-DAG: define hidden i32 @staticfunc.llvm.0
; EXPORTSTATIC-DAG: define hidden void @staticfunc2.llvm.0

; Ensure that weak alias to an imported function is correctly turned into
; a declaration.
; Also ensures that alias to a linkonce function is turned into a declaration
; and that the associated linkonce function is not in the output, as it is
; lazily linked and never referenced/materialized.
; RUN: llvm-lto -thinlto-action=import %t2.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=IMPORTGLOB1
; IMPORTGLOB1-DAG: define available_externally void @globalfunc1
; IMPORTGLOB1-DAG: declare void @weakalias
; IMPORTGLOB1-NOT: @linkoncealias
; IMPORTGLOB1-NOT: @linkoncefunc

; A strong alias is imported as an available_externally copy of its aliasee.
; IMPORTGLOB1-DAG: define available_externally void @analias
; IMPORTGLOB1-NOT: declare void @globalfunc2

; Verify that the optimizer run
; RUN: llvm-lto -thinlto-action=optimize %t2.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=OPTIMIZED
; OPTIMIZED: define i32 @main()

; Verify that the codegen run
; RUN: llvm-lto -thinlto-action=codegen %t2.bc -o - | llvm-nm -o - | FileCheck %s --check-prefix=CODEGEN
; CODEGEN: T _main

; Verify that all run together
; RUN: llvm-lto -thinlto-action=run %t2.bc  %t.bc  -exported-symbol=_main
; RUN: llvm-nm -o - < %t.bc.thinlto.o | FileCheck %s --check-prefix=ALL
; RUN: llvm-nm -o - < %t2.bc.thinlto.o | FileCheck %s --check-prefix=ALL2
; ALL: T _callfuncptr
; ALL2: T _main

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

@globalvar_in_section = global i32 1, align 4
@globalvar = global i32 1, align 4
@staticvar = internal global i32 1, align 4
@staticvar2 = internal global i32 1, align 4
@staticconstvar = internal unnamed_addr constant [2 x i32] [i32 10, i32 20], align 4
@commonvar = common global i32 0, align 4
@P = internal global void ()* null, align 8

@weakalias = weak alias void (...), bitcast (void ()* @globalfunc1 to void (...)*)
@analias = alias void (...), bitcast (void ()* @globalfunc2 to void (...)*)
@linkoncealias = alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)

define void @globalfunc1() #0 {
entry:
  ret void
}

define void @globalfunc2() #0 {
entry:
  ret void
}

define linkonce_odr void @linkoncefunc() #0 {
entry:
  ret void
}

define i32 @referencestatics(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %call = call i32 @staticfunc()
  %0 = load i32, i32* @staticvar, align 4
  %add = add nsw i32 %call, %0
  %1 = load i32, i32* %i.addr, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* @staticconstvar, i64 0, i64 %idxprom
  %2 = load i32, i32* %arrayidx, align 4
  %add1 = add nsw i32 %add, %2
  ret i32 %add1
}

define i32 @referenceglobals(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @globalfunc1()
  %0 = load i32, i32* @globalvar, align 4
  ret i32 %0
}

define i32 @referencecommon(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* @commonvar, align 4
  ret i32 %0
}

define void @setfuncptr() #0 {
entry:
  store void ()* @staticfunc2, void ()** @P, align 8
  ret void
}

define void @callfuncptr() #0 {
entry:
  %0 = load void ()*, void ()** @P, align 8
  call void %0()
  ret void
}

@weakvar = weak global i32 1, align 4
define weak void @weakfunc() #0 {
entry:
  ret void
}

define void @callweakfunc() #0 {
entry:
  call void @weakfunc()
  ret void
}

define internal i32 @staticfunc() #0 {
entry:
  ret i32 1
}

define internal void @staticfunc2() #0 {
entry:
  %0 = load i32, i32* @staticvar2, align 4
  ret void
}

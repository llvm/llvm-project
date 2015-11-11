; Do setup work for all below tests: generate bitcode and combined index
; RUN: llvm-as -function-summary %s -o %t.bc
; RUN: llvm-as -function-summary %p/Inputs/funcimport.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Ensure statics are promoted/renamed correctly from this file (all but
; constant variable need promotion).
; RUN: llvm-link %t.bc -functionindex=%t3.thinlto.bc -S | FileCheck %s --check-prefix=EXPORTSTATIC
; EXPORTSTATIC: @staticvar.llvm.1 = hidden global
; EXPORTSTATIC: @staticconstvar = internal unnamed_addr constant
; EXPORTSTATIC: @P.llvm.1 = hidden global void ()* null
; EXPORTSTATIC: define hidden i32 @staticfunc.llvm.1
; EXPORTSTATIC: define hidden void @staticfunc2.llvm.1

; Ensure that both weak alias to an imported function and strong alias to a
; non-imported function are correctly turned into declarations.
; Also ensures that alias to a linkonce function is turned into a declaration
; and that the associated linkonce function is not in the output, as it is
; lazily linked and never referenced/materialized.
; RUN: llvm-link %t2.bc -functionindex=%t3.thinlto.bc -import=globalfunc1:%t.bc -S | FileCheck %s --check-prefix=IMPORTGLOB1
; IMPORTGLOB1: define available_externally void @globalfunc1
; IMPORTGLOB1: declare void @globalfunc2
; IMPORTGLOB1: declare extern_weak void @weakalias
; IMPORTGLOB1: declare void @analias
; IMPORTGLOB1: declare void @linkoncealias
; IMPORTGLOB1-NOT: @linkoncefunc

; Ensure that weak alias to a non-imported function is correctly
; turned into a declaration, but that strong alias to an imported function
; is imported as alias.
; RUN: llvm-link %t2.bc -functionindex=%t3.thinlto.bc -import=globalfunc2:%t.bc -S | FileCheck %s --check-prefix=IMPORTGLOB2
; IMPORTGLOB2: @analias = alias void (...), bitcast (void ()* @globalfunc2
; IMPORTGLOB2: declare void @globalfunc1
; IMPORTGLOB2: define available_externally void @globalfunc2
; IMPORTGLOB2: declare extern_weak void @weakalias

; Ensure that strong alias imported in second pass of importing ends up
; as an alias.
; RUN: llvm-link %t2.bc -functionindex=%t3.thinlto.bc -import=globalfunc1:%t.bc -import=globalfunc2:%t.bc -S | FileCheck %s --check-prefix=IMPORTGLOB3
; IMPORTGLOB3: @analias = alias void (...), bitcast (void ()* @globalfunc2
; IMPORTGLOB3: define available_externally void @globalfunc1
; IMPORTGLOB3: define available_externally void @globalfunc2
; IMPORTGLOB3: declare extern_weak void @weakalias

; Ensure that strong alias imported in first pass of importing ends up
; as an alias, and that seeing the alias definition during a second inlining
; pass is handled correctly.
; RUN: llvm-link %t2.bc -functionindex=%t3.thinlto.bc -import=globalfunc2:%t.bc -import=globalfunc1:%t.bc -S | FileCheck %s --check-prefix=IMPORTGLOB4
; IMPORTGLOB4: @analias = alias void (...), bitcast (void ()* @globalfunc2
; IMPORTGLOB4: define available_externally void @globalfunc2
; IMPORTGLOB4: define available_externally void @globalfunc1
; IMPORTGLOB4: declare extern_weak void @weakalias

; Ensure that imported static variable and function references are correctly
; promoted and renamed (including static constant variable).
; RUN: llvm-link %t2.bc -functionindex=%t3.thinlto.bc -import=referencestatics:%t.bc -S | FileCheck %s --check-prefix=IMPORTSTATIC
; IMPORTSTATIC: @staticvar.llvm.1 = available_externally hidden global
; IMPORTSTATIC: @staticconstvar.llvm.1 = internal unnamed_addr constant
; IMPORTSTATIC: define available_externally i32 @referencestatics
; IMPORTSTATIC: %call = call i32 @staticfunc.llvm.1
; IMPORTSTATIC: %0 = load i32, i32* @staticvar.llvm.1
; IMPORTSTATIC: declare hidden i32 @staticfunc.llvm.1

; Ensure that imported global (external) function and variable references
; are handled correctly (including referenced variable imported as
; available_externally definition)
; RUN: llvm-link %t2.bc -functionindex=%t3.thinlto.bc -import=referenceglobals:%t.bc -S | FileCheck %s --check-prefix=IMPORTGLOBALS
; IMPORTGLOBALS: @globalvar = available_externally global
; IMPORTGLOBALS: declare void @globalfunc1()
; IMPORTGLOBALS: define available_externally i32 @referenceglobals

; Ensure that common variable correctly imported as common defition.
; RUN: llvm-link %t2.bc -functionindex=%t3.thinlto.bc -import=referencecommon:%t.bc -S | FileCheck %s --check-prefix=IMPORTCOMMON
; IMPORTCOMMON: @commonvar = common global
; IMPORTCOMMON: define available_externally i32 @referencecommon

; Ensure that imported static function pointer correctly promoted and renamed.
; RUN: llvm-link %t2.bc -functionindex=%t3.thinlto.bc -import=callfuncptr:%t.bc -S | FileCheck %s --check-prefix=IMPORTFUNCPTR
; IMPORTFUNCPTR: @P.llvm.1 = available_externally hidden global void ()* null
; IMPORTFUNCPTR: define available_externally void @callfuncptr
; IMPORTFUNCPTR: %0 = load void ()*, void ()** @P.llvm.1

; Ensure that imported weak function reference/definition handled properly.
; Imported weak_any definition should be skipped with warning, and imported
; reference should turned into an external_weak declaration.
; RUN: llvm-link %t2.bc -functionindex=%t3.thinlto.bc -import=callweakfunc:%t.bc -import=weakfunc:%t.bc -S 2>&1 | FileCheck %s --check-prefix=IMPORTWEAKFUNC
; IMPORTWEAKFUNC: Ignoring import request for weak-any function weakfunc
; IMPORTWEAKFUNC: @weakvar = extern_weak global i32, align 4
; IMPORTWEAKFUNC: declare extern_weak void @weakfunc
; IMPORTWEAKFUNC: define available_externally void @callweakfunc

@globalvar = global i32 1, align 4
@staticvar = internal global i32 1, align 4
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
  ret void
}

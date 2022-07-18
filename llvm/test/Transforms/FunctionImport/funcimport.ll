; REQUIRES: x86-registered-target

; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport.ll -o %t2.bc
; RUN: llvm-lto -thinlto -print-summary-global-ids -o %t3 %t.bc %t2.bc 2>&1 | FileCheck %s --check-prefix=GUID

; Do the import now
; RUN: opt -passes=function-import -stats -print-imports -enable-import-metadata -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=INSTLIMDEF
; Try again with new pass manager
; RUN: opt -passes='function-import' -stats -print-imports -enable-import-metadata -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=INSTLIMDEF
; RUN: opt -passes='function-import' -debug-only=function-import -enable-import-metadata -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s --check-prefix=DUMP
; "-stats" and "-debug-only" require +Asserts.
; REQUIRES: asserts

; Test import with smaller instruction limit
; RUN: opt -passes=function-import -enable-import-metadata  -summary-file %t3.thinlto.bc %t.bc -import-instr-limit=5 -S | FileCheck %s --check-prefix=CHECK --check-prefix=INSTLIM5
; INSTLIM5-NOT: @staticfunc.llvm.

; Test force import all
; RUN: llvm-lto -thinlto-action=run -force-import-all %t.bc %t2.bc 2>&1 \
; RUN:  | FileCheck %s --check-prefix=IMPORTALL
; IMPORTALL-DAG: Error importing module: Failed to import function weakalias due to InterposableLinkage

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define i32 @main() #0 {
entry:
  call void (...) @weakalias()
  call void (...) @analias()
  call void (...) @linkoncealias()
  %call = call i32 (...) @referencestatics()
  %call1 = call i32 (...) @referenceglobals()
  %call2 = call i32 (...) @referencecommon()
  call void (...) @setfuncptr()
  call void (...) @callfuncptr()
  call void (...) @weakfunc()
  call void (...) @linkoncefunc2()
  call void (...) @referencelargelinkonce()
  call void (...) @variadic_no_va_start()
  call void (...) @variadic_va_start()
  ret i32 0
}

; Won't import weak alias
; CHECK-DAG: declare void @weakalias
declare void @weakalias(...) #1

; External alias imported as available_externally copy of aliasee
; CHECK-DAG: define available_externally void @analias
declare void @analias(...) #1

; External alias imported as available_externally copy of aliasee
; (linkoncealias is an external alias to a linkonce_odr)
declare void @linkoncealias(...) #1
; CHECK-DAG: define available_externally void @linkoncealias()

; INSTLIMDEF-DAG: Import referencestatics
; INSTLIMDEF-DAG: define available_externally i32 @referencestatics(i32 %i) !thinlto_src_module !0 {
; INSTLIM5-DAG: declare i32 @referencestatics(...)
declare i32 @referencestatics(...) #1

; The import of referencestatics will expose call to staticfunc that
; should in turn be imported as a promoted/renamed and hidden function.
; Ensure that the call is to the properly-renamed function.
; INSTLIMDEF-DAG: Import staticfunc
; INSTLIMDEF-DAG: %call = call i32 @staticfunc.llvm.
; INSTLIMDEF-DAG: define available_externally hidden i32 @staticfunc.llvm.{{.*}} !thinlto_src_module !0 {

; INSTLIMDEF-DAG: Import referenceglobals
; CHECK-DAG: define available_externally i32 @referenceglobals(i32 %i) !thinlto_src_module !0 {
declare i32 @referenceglobals(...) #1

; The import of referenceglobals will expose call to globalfunc1 that
; should in turn be imported.
; INSTLIMDEF-DAG: Import globalfunc1
; CHECK-DAG: define available_externally void @globalfunc1() !thinlto_src_module !0

; INSTLIMDEF-DAG: Import referencecommon
; CHECK-DAG: define available_externally i32 @referencecommon(i32 %i) !thinlto_src_module !0 {
declare i32 @referencecommon(...) #1

; INSTLIMDEF-DAG: Import setfuncptr
; CHECK-DAG: define available_externally void @setfuncptr() !thinlto_src_module !0 {
declare void @setfuncptr(...) #1

; INSTLIMDEF-DAG: Import callfuncptr
; CHECK-DAG: define available_externally void @callfuncptr() !thinlto_src_module !0 {
declare void @callfuncptr(...) #1

; Ensure that all uses of local variable @P which has used in setfuncptr
; and callfuncptr are to the same promoted/renamed global.
; CHECK-DAG: @P.llvm.{{.*}} = available_externally hidden global void ()* null
; CHECK-DAG: %0 = load void ()*, void ()** @P.llvm.
; CHECK-DAG: store void ()* @staticfunc2.llvm.{{.*}}, void ()** @P.llvm.

; Ensure that @referencelargelinkonce definition is pulled in, but later we
; also check that the linkonceodr function is not.
; CHECK-DAG: define available_externally void @referencelargelinkonce() !thinlto_src_module !0 {
; INSTLIM5-DAG: declare void @linkonceodr()
declare void @referencelargelinkonce(...)

; Won't import weak func
; CHECK-DAG: declare void @weakfunc(...)
declare void @weakfunc(...) #1

; Won't import linkonce func
; CHECK-DAG: declare void @linkoncefunc2(...)
declare void @linkoncefunc2(...) #1

; INSTLIMDEF-DAG: Import funcwithpersonality
; INSTLIMDEF-DAG: define available_externally hidden void @funcwithpersonality.llvm.{{.*}}() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !thinlto_src_module !0 {
; INSTLIM5-DAG: declare hidden void @funcwithpersonality.llvm.{{.*}}()

; We can import variadic functions without a va_start, since the inliner
; can handle them.
; INSTLIMDEF-DAG: Import variadic_no_va_start
; CHECK-DAG: define available_externally void @variadic_no_va_start(...) !thinlto_src_module !0 {
declare void @variadic_no_va_start(...)

; We can import variadic functions with a va_start, since the inliner
; can sometimes handle them.
; CHECK-DAG: define available_externally void @variadic_va_start(...)
declare void @variadic_va_start(...)

; INSTLIMDEF-DAG: Import globalfunc2
; INSTLIMDEF-DAG: 15 function-import - Number of functions imported
; INSTLIMDEF-DAG: 4 function-import - Number of global variables imported

; CHECK-DAG: !0 = !{!"{{.*}}/Inputs/funcimport.ll"}

; The actual GUID values will depend on path to test.
; GUID-DAG: GUID {{.*}} is weakalias
; GUID-DAG: GUID {{.*}} is referenceglobals
; GUID-DAG: GUID {{.*}} is weakfunc
; GUID-DAG: GUID {{.*}} is main
; GUID-DAG: GUID {{.*}} is referencecommon
; GUID-DAG: GUID {{.*}} is analias
; GUID-DAG: GUID {{.*}} is referencestatics
; GUID-DAG: GUID {{.*}} is linkoncealias
; GUID-DAG: GUID {{.*}} is setfuncptr
; GUID-DAG: GUID {{.*}} is callfuncptr
; GUID-DAG: GUID {{.*}} is funcwithpersonality
; GUID-DAG: GUID {{.*}} is setfuncptr
; GUID-DAG: GUID {{.*}} is staticfunc2
; GUID-DAG: GUID {{.*}} is __gxx_personality_v0
; GUID-DAG: GUID {{.*}} is referencestatics
; GUID-DAG: GUID {{.*}} is globalfunc1
; GUID-DAG: GUID {{.*}} is globalfunc2
; GUID-DAG: GUID {{.*}} is P
; GUID-DAG: GUID {{.*}} is staticvar
; GUID-DAG: GUID {{.*}} is commonvar
; GUID-DAG: GUID {{.*}} is weakalias
; GUID-DAG: GUID {{.*}} is staticfunc
; GUID-DAG: GUID {{.*}} is weakfunc
; GUID-DAG: GUID {{.*}} is referenceglobals
; GUID-DAG: GUID {{.*}} is weakvar
; GUID-DAG: GUID {{.*}} is staticconstvar
; GUID-DAG: GUID {{.*}} is analias
; GUID-DAG: GUID {{.*}} is globalvar
; GUID-DAG: GUID {{.*}} is referencecommon
; GUID-DAG: GUID {{.*}} is linkoncealias
; GUID-DAG: GUID {{.*}} is callfuncptr
; GUID-DAG: GUID {{.*}} is linkoncefunc

; DUMP:       Module [[M1:.*]] imports from 1 module
; DUMP-NEXT:  15 functions imported from [[M2:.*]]
; DUMP-NEXT:  4 vars imported from [[M2]]
; DUMP:       Imported 15 functions for Module [[M1]]
; DUMP-NEXT:  Imported 4 global variables for Module [[M1]]

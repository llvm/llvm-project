; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport.ll -o %t2.bc
; RUN: llvm-lto -thinlto -print-summary-global-ids -o %t3 %t.bc %t2.bc 2>&1 | FileCheck %s --check-prefix=GUID

; Do the import now
; RUN: opt -disable-force-link-odr -function-import -stats -print-imports -enable-import-metadata -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=INSTLIMDEF
; Try again with new pass manager
; RUN: opt -disable-force-link-odr -passes='function-import' -stats -print-imports -enable-import-metadata -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=INSTLIMDEF
; "-stats" requires +Asserts.
; REQUIRES: asserts

; Test import with smaller instruction limit
; RUN: opt -disable-force-link-odr -function-import -enable-import-metadata  -summary-file %t3.thinlto.bc %t.bc -import-instr-limit=5 -S | FileCheck %s --check-prefix=CHECK --check-prefix=INSTLIM5
; INSTLIM5-NOT: @staticfunc.llvm.

; Test import with smaller instruction limit and without the -disable-force-link-odr
; RUN: opt -function-import -summary-file %t3.thinlto.bc %t.bc -import-instr-limit=5 -S | FileCheck %s --check-prefix=INSTLIM5ODR
; INSTLIM5ODR: define linkonce_odr void @linkonceodr() {


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
  call void (...) @variadic()
  ret i32 0
}

; Won't import weak alias
; CHECK-DAG: declare void @weakalias
declare void @weakalias(...) #1

; Cannot create an alias to available_externally
; CHECK-DAG: declare void @analias
declare void @analias(...) #1

; FIXME: Add this checking back when follow on fix to add alias summary
; records is committed.
; Aliases import the aliasee function
declare void @linkoncealias(...) #1

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
; CHECK-DAG: @P.llvm.{{.*}} = external hidden global void ()*
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

; CHECK-DAG: declare void @variadic(...)
declare void @variadic(...)

; INSTLIMDEF-DAG: Import globalfunc2
; INSTLIMDEF-DAG: 13 function-import - Number of functions imported
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

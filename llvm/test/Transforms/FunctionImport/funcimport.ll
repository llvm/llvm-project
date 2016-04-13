; Do setup work for all below tests: generate bitcode and combined index
; RUN: llvm-as -module-summary %s -o %t.bc
; RUN: llvm-as -module-summary %p/Inputs/funcimport.ll -o %t2.bc
; RUN: llvm-lto -thinlto -print-summary-global-ids -o %t3 %t.bc %t2.bc 2>&1 | FileCheck %s --check-prefix=GUID

; Do the import now
; RUN: opt -function-import -stats -print-imports -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=INSTLIMDEF

; Test import with smaller instruction limit
; RUN: opt -function-import -summary-file %t3.thinlto.bc %t.bc -import-instr-limit=5 -S | FileCheck %s --check-prefix=CHECK --check-prefix=INSTLIM5
; INSTLIM5-NOT: @staticfunc.llvm.2

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
  ret i32 0
}

; Won't import weak alias
; CHECK-DAG: declare void @weakalias
declare void @weakalias(...) #1

; Cannot create an alias to available_externally
; CHECK-DAG: declare void @analias
declare void @analias(...) #1

; Aliases import the aliasee function
declare void @linkoncealias(...) #1
; INSTLIMDEF-DAG: Import linkoncealias
; INSTLIMDEF-DAG: Import linkoncefunc
; CHECK-DAG: define linkonce_odr void @linkoncefunc()
; CHECK-DAG: @linkoncealias = alias void (...), bitcast (void ()* @linkoncefunc to void (...)*

; INSTLIMDEF-DAG: Import referencestatics
; INSTLIMDEF-DAG: define available_externally i32 @referencestatics(i32 %i)
; INSTLIM5-DAG: declare i32 @referencestatics(...)
declare i32 @referencestatics(...) #1

; The import of referencestatics will expose call to staticfunc that
; should in turn be imported as a promoted/renamed and hidden function.
; Ensure that the call is to the properly-renamed function.
; INSTLIMDEF-DAG: Import staticfunc
; INSTLIMDEF-DAG: %call = call i32 @staticfunc.llvm.2()
; INSTLIMDEF-DAG: define available_externally hidden i32 @staticfunc.llvm.2()

; INSTLIMDEF-DAG: Import referenceglobals
; CHECK-DAG: define available_externally i32 @referenceglobals(i32 %i)
declare i32 @referenceglobals(...) #1

; The import of referenceglobals will expose call to globalfunc1 that
; should in turn be imported.
; INSTLIMDEF-DAG: Import globalfunc1
; CHECK-DAG: define available_externally void @globalfunc1()

; INSTLIMDEF-DAG: Import referencecommon
; CHECK-DAG: define available_externally i32 @referencecommon(i32 %i)
declare i32 @referencecommon(...) #1

; INSTLIMDEF-DAG: Import setfuncptr
; CHECK-DAG: define available_externally void @setfuncptr()
declare void @setfuncptr(...) #1

; INSTLIMDEF-DAG: Import callfuncptr
; CHECK-DAG: define available_externally void @callfuncptr()
declare void @callfuncptr(...) #1

; Ensure that all uses of local variable @P which has used in setfuncptr
; and callfuncptr are to the same promoted/renamed global.
; CHECK-DAG: @P.llvm.2 = external hidden global void ()*
; CHECK-DAG: %0 = load void ()*, void ()** @P.llvm.2,
; CHECK-DAG: store void ()* @staticfunc2.llvm.2, void ()** @P.llvm.2,

; Won't import weak func
; CHECK-DAG: declare void @weakfunc(...)
declare void @weakfunc(...) #1

; INSTLIMDEF-DAG: Import funcwithpersonality
; INSTLIMDEF-DAG: define available_externally hidden void @funcwithpersonality.llvm.2() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; INSTLIM5-DAG: declare hidden void @funcwithpersonality.llvm.2()

; INSTLIMDEF-DAG: Import globalfunc2
; INSTLIMDEF-DAG: 11 function-import - Number of functions imported

; The GUID values should be stable unless the global identifier scheme
; is changed.
; GUID-DAG: GUID 18438612909910340889 is weakalias
; GUID-DAG: GUID 10419849736188691048 is referenceglobals
; GUID-DAG: GUID 9779356664709680872 is weakfunc
; GUID-DAG: GUID 15822663052811949562 is main
; GUID-DAG: GUID 1794834399867852914 is referencecommon
; GUID-DAG: GUID 12695095382722328222 is analias
; GUID-DAG: GUID 11460932053523480915 is referencestatics
; GUID-DAG: GUID 17082925359558765443 is linkoncealias
; GUID-DAG: GUID 16956293738471118660 is setfuncptr
; GUID-DAG: GUID 800887449839734011 is callfuncptr
; GUID-DAG: GUID 12108029313449967842 is funcwithpersonality
; GUID-DAG: GUID 16956293738471118660 is setfuncptr
; GUID-DAG: GUID 15894399990123115248 is staticfunc2
; GUID-DAG: GUID 1233668236132380018 is __gxx_personality_v0
; GUID-DAG: GUID 11460932053523480915 is referencestatics
; GUID-DAG: GUID 8332887114342655934 is globalfunc1
; GUID-DAG: GUID 2602152165807499502 is globalfunc2
; GUID-DAG: GUID 9342344237287280920 is P
; GUID-DAG: GUID 17578217388980876465 is staticvar
; GUID-DAG: GUID 3013670425691502549 is commonvar
; GUID-DAG: GUID 18438612909910340889 is weakalias
; GUID-DAG: GUID 13921022463002872889 is staticfunc
; GUID-DAG: GUID 9779356664709680872 is weakfunc
; GUID-DAG: GUID 10419849736188691048 is referenceglobals
; GUID-DAG: GUID 8769477226392140800 is weakvar
; GUID-DAG: GUID 16489816843137310249 is staticconstvar
; GUID-DAG: GUID 12695095382722328222 is analias
; GUID-DAG: GUID 12887606300320728018 is globalvar
; GUID-DAG: GUID 1794834399867852914 is referencecommon
; GUID-DAG: GUID 17082925359558765443 is linkoncealias
; GUID-DAG: GUID 800887449839734011 is callfuncptr
; GUID-DAG: GUID 7812846502172333492 is linkoncefunc

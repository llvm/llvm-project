; Do setup work for all below tests: generate bitcode and combined index
; RUN: llvm-as -function-summary %s -o %t.bc
; RUN: llvm-as -function-summary %p/Inputs/funcimport.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Do the import now
; RUN: opt -function-import -summary-file %t3.thinlto.bc %s -S | FileCheck %s --check-prefix=CHECK --check-prefix=INSTLIMDEF

; Test import with smaller instruction limit
; RUN: opt -function-import -summary-file %t3.thinlto.bc %s -import-instr-limit=5 -S | FileCheck %s --check-prefix=CHECK --check-prefix=INSTLIM5
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
; CHECK-DAG: define linkonce_odr void @linkoncefunc()
; CHECK-DAG: @linkoncealias = alias void (...), bitcast (void ()* @linkoncefunc to void (...)*

; INSTLIMDEF-DAG: define available_externally i32 @referencestatics(i32 %i)
; INSTLIM5-DAG: declare i32 @referencestatics(...)
declare i32 @referencestatics(...) #1

; The import of referencestatics will expose call to staticfunc that
; should in turn be imported as a promoted/renamed and hidden function.
; Ensure that the call is to the properly-renamed function.
; INSTLIMDEF-DAG: %call = call i32 @staticfunc.llvm.2()
; INSTLIMDEF-DAG: define available_externally hidden i32 @staticfunc.llvm.2()

; CHECK-DAG: define available_externally i32 @referenceglobals(i32 %i)
declare i32 @referenceglobals(...) #1

; The import of referenceglobals will expose call to globalfunc1 that
; should in turn be imported.
; CHECK-DAG: define available_externally void @globalfunc1()

; CHECK-DAG: define available_externally i32 @referencecommon(i32 %i)
declare i32 @referencecommon(...) #1

; CHECK-DAG: define available_externally void @setfuncptr()
declare void @setfuncptr(...) #1

; CHECK-DAG: define available_externally void @callfuncptr()
declare void @callfuncptr(...) #1

; Ensure that all uses of local variable @P which has used in setfuncptr
; and callfuncptr are to the same promoted/renamed global.
; CHECK-DAG: @P.llvm.2 = available_externally hidden global void ()* null
; CHECK-DAG: %0 = load void ()*, void ()** @P.llvm.2,
; CHECK-DAG: store void ()* @staticfunc2.llvm.2, void ()** @P.llvm.2,

; Won't import weak func
; CHECK-DAG: declare void @weakfunc(...)
declare void @weakfunc(...) #1


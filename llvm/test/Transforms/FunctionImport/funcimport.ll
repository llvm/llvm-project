; Do setup work for all below tests: generate bitcode and combined index
; RUN: llvm-as -function-summary %s -o %t.bc
; RUN: llvm-as -function-summary %p/Inputs/funcimport.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Do the import now
; RUN: opt -function-import -summary-file %t3.thinlto.bc %s -S | FileCheck %s

define i32 @main() #0 {
entry:
  call void (...) @weakalias()
  call void (...) @analias()
  %call = call i32 (...) @referencestatics()
  %call1 = call i32 (...) @referenceglobals()
  %call2 = call i32 (...) @referencecommon()
  call void (...) @setfuncptr()
  call void (...) @callfuncptr()
  call void (...) @callweakfunc()
  ret i32 0
}

; Won't import alias
declare void @weakalias(...) #1
declare void @analias(...) #1

; CHECK-DAG: define available_externally i32 @referencestatics(i32 %i)
declare i32 @referencestatics(...) #1

; CHECK-DAG: define available_externally i32 @referenceglobals(i32 %i)
declare i32 @referenceglobals(...) #1

; CHECK-DAG: define available_externally i32 @referencecommon(i32 %i)
declare i32 @referencecommon(...) #1

; CHECK-DAG: define available_externally void @setfuncptr()
declare void @setfuncptr(...) #1

; CHECK-DAG: define available_externally void @callfuncptr()
declare void @callfuncptr(...) #1

; Won't import weak func
declare void @callweakfunc(...) #1

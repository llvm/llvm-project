; RUN: opt < %s -inline -inline-caller-superset-tli=true -S | FileCheck %s --check-prefixes=COMMON
; RUN: opt < %s -passes='cgscc(inline)' -inline-caller-superset-tli=true -S | FileCheck %s --check-prefixes=COMMON
; RUN: opt < %s -inline -inline-caller-superset-tli=false -S | FileCheck %s --check-prefixes=NOSUPERSET,COMMON
; RUN: opt < %s -passes='cgscc(inline)' -inline-caller-superset-tli=false -S | FileCheck %s --check-prefixes=NOSUPERSET,COMMON



define i32 @callee_svml(i8 %X) #0 {
entry:
  ret i32 1
}

define i32 @callee_massv(i8 %X) #1 {
entry:
  ret i32 1
}

define i32 @callee_nolibrary(i8 %X) {
entry:
  ret i32 1
}

define i32 @caller_svml() #0 {
; COMMON-LABEL: define i32 @caller_svml()
entry:
  %rslt = call i32 @callee_massv(i8 123)
; COMMON: call i32 @callee_massv
  %tmp1 = call i32 @callee_nolibrary(i8 123)
; NOSUPERSET: call i32 @callee_nolibrary
  %tmp2 = call i32 @callee_svml(i8 123)
; COMMON-NOT: call
  ret i32 %rslt
}

define i32 @caller_nolibrary() {
; COMMON-LABEL: define i32 @caller_nolibrary()
entry:
  %rslt = call i32 @callee_svml(i8 123)
; COMMON: call i32 @callee_svml
  %tmp1 = call i32 @callee_massv(i8 123)
; COMMON: call i32 @callee_massv
  %tmp2 = call i32 @callee_nolibrary(i8 123)
; COMMON-NOT: call
  ret i32 %rslt
}

attributes #0 = { "veclib"="SVML" }
attributes #1 = { "veclib"="MASSV" }
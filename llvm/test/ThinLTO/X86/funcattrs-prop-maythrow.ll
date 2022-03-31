; For instructions explicitly defined as mayThrow, make sure they prevent nounwind propagation
; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/main.ll -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: opt -thinlto-bc %t/callees.ll -thin-link-bitcode-file=%t2.thinlink.bc -o %t2.bc
; RUN: llvm-lto2 run -disable-thinlto-funcattrs=0 %t1.bc %t2.bc -o %t.o -r %t1.bc,caller,px -r %t1.bc,caller1,px -r %t1.bc,caller2,px -r %t1.bc,caller_nounwind,px  \
; RUN:               -r %t1.bc,cleanupret,l -r %t1.bc,catchret,l -r %t1.bc,resume,l -r %t1.bc,cleanupret_nounwind,l \
; RUN:               -r %t2.bc,cleanupret,px -r %t2.bc,catchret,px -r %t2.bc,resume,px -r %t2.bc,cleanupret_nounwind,px -r %t2.bc,nonThrowing,px -r %t2.bc,__gxx_personality_v0,px -save-temps
; RUN: llvm-dis -o - %t2.bc | FileCheck %s --check-prefix=SUMMARY
; RUN: llvm-dis -o - %t.o.1.3.import.bc | FileCheck %s

;--- main.ll
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @cleanupret()
declare void @catchret()
declare void @resume()

; Functions can have mayThrow instructions but also be marked noUnwind
; if they have terminate semantics (e.g. noexcept). In such cases
; propagation trusts the original noUnwind value in the function summary
declare void @cleanupret_nounwind()

; CHECK: define void @caller() [[ATTR_MAYTHROW:#[0-9]+]]
define void @caller() {
  call void @cleanupret()
  ret void
}

; CHECK: define void @caller1() [[ATTR_MAYTHROW:#[0-9]+]]
define void @caller1() {
  call void @catchret()
  ret void
}

; CHECK: define void @caller2() [[ATTR_MAYTHROW:#[0-9]+]]
define void @caller2() {
  call void @resume()
  ret void
}

; CHECK: define void @caller_nounwind() [[ATTR_NOUNWIND:#[0-9]+]]
define void @caller_nounwind() {
    call void @cleanupret_nounwind()
    ret void
}

; CHECK-DAG: attributes [[ATTR_NOUNWIND]] = { norecurse nounwind }
; CHECK-DAG: attributes [[ATTR_MAYTHROW]] = { norecurse }

; SUMMARY-DAG: = gv: (name: "cleanupret", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 4, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 0, mayThrow: 1, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^{{.*}})), refs: (^{{.*}}))))
; SUMMARY-DAG: = gv: (name: "resume", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 4, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 0, mayThrow: 1, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^{{.*}})), refs: (^{{.*}}))))
; SUMMARY-DAG: = gv: (name: "catchret", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 5, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 0, mayThrow: 1, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^{{.*}})), refs: (^{{.*}}))))

;--- callees.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
attributes #0 = { nounwind }

define void @nonThrowing() #0 {
    ret void
}

declare i32 @__gxx_personality_v0(...)

define void @cleanupret() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @nonThrowing()
          to label %exit unwind label %pad
pad:
  %cp = cleanuppad within none [i7 4]
  cleanupret from %cp unwind to caller
exit:
  ret void
}

define void @catchret() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @nonThrowing()
          to label %exit unwind label %pad
pad:
  %cs1 = catchswitch within none [label %catch] unwind to caller
catch:
  %cp = catchpad within %cs1 [i7 4]
  catchret from %cp to label %exit
exit:
  ret void
}

define void @resume() uwtable optsize ssp personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @nonThrowing()
          to label %try.cont unwind label %lpad

try.cont:                                         ; preds = %entry, %invoke.cont4
  ret void

lpad:                                             ; preds = %entry
  %exn = landingpad {i8*, i32}
           cleanup
  resume { i8*, i32 } %exn
}

define void @cleanupret_nounwind() #0 personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @nonThrowing()
          to label %exit unwind label %pad
pad:
  %cp = cleanuppad within none [i7 4]
  cleanupret from %cp unwind to caller
exit:
  ret void
}

attributes #0 = { nounwind }
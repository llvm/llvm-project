; REQUIRES: asserts

; RUN: not --crash opt -passes=crash-function %s 2> %t
; RUN: FileCheck --input-file=%t %s

; CHECK:      Stack dump:
; CHECK-NEXT: 0.  Program arguments:
; CHECK-NEXT: 1.  Running pass 'ModuleToFunctionPassAdaptor' on module
; CHECK-NEXT: 2.  Running pass 'PassManager<{{.*}}llvm::Function{{.*}}>' on function '@foo'
; CHECK-NEXT: 3.  Running pass 'CrashingFunctionPass' on function '@foo'

define void @foo() {
  ret void
}

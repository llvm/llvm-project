; REQUIRES: asserts

; RUN: not --crash opt -passes=crash-loop %s 2> %t
; RUN: FileCheck --input-file=%t %s

; CHECK:      Stack dump:
; CHECK-NEXT: 0.  Program arguments:
; CHECK-NEXT: 1.  Running pass 'ModuleToFunctionPassAdaptor' on module
; CHECK-NEXT: 2.  Running pass 'PassManager<{{.*}}llvm::Function{{.*}}>' on function '@foo'
; CHECK-NEXT: 3.  Running pass 'FunctionToLoopPassAdaptor' on function '@foo'
; CHECK-NEXT: 4.  Running pass 'PassManager<{{.*}}llvm::Loop,{{.+}}>' on loop 'loop.header'.
; CHECK-NEXT: 5.  Running pass 'CrashingLoopPass' on loop 'loop.header'.

define void @foo() {
entry:
  br label %loop.header

loop.header:
  br label %loop.header
}

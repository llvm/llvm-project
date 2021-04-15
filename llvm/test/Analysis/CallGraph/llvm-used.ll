; RUN: opt < %s -print-callgraph -disable-output 2>&1 | FileCheck %s

; The test will report used1 and used2 functions as used on the grounds
; of llvm.*.used references. Passing IgnoreLLVMUsed = true into the
; Function::hasAddressTaken() in the CallGraph::addToCallGraph() has to
; change their uses to zero.

; CHECK: Call graph node <<null function>><<{{.*}}>>  #uses=0
; CHECK-NEXT:  CS<None> calls function 'used1'
; CHECK-NEXT:  CS<None> calls function 'used2'
; CHECK-NEXT:  CS<None> calls function 'unused'
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'unused'<<{{.*}}>>  #uses=1
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'used1'<<{{.*}}>>  #uses=1
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'used2'<<{{.*}}>>  #uses=1
; CHECK-EMPTY:

@llvm.used = appending global [1 x i8*] [i8* bitcast (void ()* @used1 to i8*)]
@llvm.compiler.used = appending global [1 x void()*] [void ()* @used2]
@array = appending global [1 x i8*] [i8* bitcast (void ()* @unused to i8*)]

define internal void @used1() {
entry:
  ret void
}

define internal void @used2() {
entry:
  ret void
}

define internal void @unused() {
entry:
  ret void
}

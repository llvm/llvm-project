; RUN: opt < %s -print-callgraph -disable-output 2>&1 | FileCheck %s
; CHECK: Call graph node <<null function>><<{{.*}}>>  #uses=0
; CHECK-NEXT:  CS<None> calls function 'unused'
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'unused'<<{{.*}}>>  #uses=1
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'used1'<<{{.*}}>>  #uses=0
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'used2'<<{{.*}}>>  #uses=0
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

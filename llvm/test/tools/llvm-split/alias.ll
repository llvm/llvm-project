; RUN: llvm-split --opaque-pointers=0 -o %t %s
; RUN: llvm-dis --opaque-pointers=0 -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis --opaque-pointers=0 -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0-DAG: @afoo = alias [2 x i8*], [2 x i8*]* @foo
; CHECK1-DAG: @afoo = external global [2 x i8*]
@afoo = alias [2 x i8*], [2 x i8*]* @foo

; CHECK0-DAG: declare void @abar()
; CHECK1-DAG: @abar = alias void (), void ()* @bar
@abar = alias void (), void ()* @bar

@foo = global [2 x i8*] [i8* bitcast (void ()* @bar to i8*), i8* bitcast (void ()* @abar to i8*)]

define void @bar() {
  store [2 x i8*] zeroinitializer, [2 x i8*]* @foo
  store [2 x i8*] zeroinitializer, [2 x i8*]* @afoo
  ret void
}

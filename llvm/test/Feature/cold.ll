; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: @fun() #0
define void @fun() #0 {
  ret void
}

; CHECK: attributes #0 = { cold }
attributes #0 = { cold }

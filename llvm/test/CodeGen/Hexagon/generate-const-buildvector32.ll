; RUN: llc -march=hexagon -mtriple=hexagon < %s | FileCheck %s

; CHECK: r{{[0-9]+}} = ##673059850

define dso_local i32 @main() #0 {
entry:
  %a = alloca <4 x i8>, align 4
  store <4 x i8> <i8 10, i8 20, i8 30, i8 40>, <4 x i8>* %a, align 4
  ret i32 0
}


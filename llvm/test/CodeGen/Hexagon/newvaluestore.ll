; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we generate new value store.

@i = global i32 0, align 4

define i32 @main(i32 %x, i32* %p) nounwind {
entry:
; CHECK: memw(r{{[0-9]+}}+#{{[0-9]+}}) = r{{[0-9]+}}.new
  %t0 = load i32, i32* @i, align 4
  store i32 %t0, i32* %p, align 4
  ret i32 %x
}


; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; Check that we generate dual stores in one packet in V4

; CHECK: memw(r{{[0-9]+}} + #{{[0-9]+}}) = r{{[0-9]+}}
; CHECK-NEXT: memw(r{{[0-9]+}} + #{{[0-9]+}}) = r{{[0-9]+}}
; CHECK-NEXT: }

@Reg = global i32 0, align 4
define i32 @main() nounwind {
entry:
  %number= alloca i32, align 4
  store i32 500000, i32* %number, align 4
  %number1= alloca i32, align 4
  store i32 100000, i32* %number1, align 4
  ret i32 0
}


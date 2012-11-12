; RUN: llc -march=hexagon -mcpu=hexagonv4 -disable-hexagon-misched < %s | FileCheck %s
; Check that we generate new value store packet in V4

@i = global i32 0, align 4
@j = global i32 10, align 4
@k = global i32 100, align 4

define i32 @main() nounwind {
entry:
; CHECK: memw(r{{[0-9]+}} + #{{[0-9]+}}) = r{{[0-9]+}}.new
  %number1 = alloca i32, align 4
  %number2 = alloca i32, align 4
  %number3 = alloca i32, align 4
  %0 = load i32 * @i, align 4
  store i32 %0, i32* %number1, align 4
  %1 = load i32 * @j, align 4
  store i32 %1, i32* %number2, align 4
  %2 = load i32 * @k, align 4
  store i32 %2, i32* %number3, align 4
  ret i32 %0
}


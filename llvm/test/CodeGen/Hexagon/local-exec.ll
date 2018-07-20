; RUN: llc -march=hexagon -O0 < %s | FileCheck %s

target triple = "hexagon-unknown--elf"

@g0 = internal thread_local(localexec) global i32 0, align 4
@g1 = internal thread_local(localexec) global i32 0, align 4
; CHECK: ##g0@{{TPREL|tprel}}
; CHECK: ##g1@{{TPREL|tprel}}

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32*, align 4
  store i32 0, i32* %v0
  store i32* @g0, i32** %v1, align 4
  %v2 = load i32, i32* @g1, align 4
  %v3 = load i32*, i32** %v1, align 4
  store i32 %v2, i32* %v3, align 4
  ret i32 0
}

attributes #0 = { nounwind }

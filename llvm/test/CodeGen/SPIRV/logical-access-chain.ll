; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s

define void @main() #1 {
entry:
  %0 = alloca <2 x i8>, align 4
  %1 = getelementptr i8, ptr %0, i32 0

  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }


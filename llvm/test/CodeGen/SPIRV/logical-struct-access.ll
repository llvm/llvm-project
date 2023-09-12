; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s

%A = type {
  i32,
  i32
}

%B = type {
  %A,
  i32,
  %A
}

define void @main() #1 {
entry:
  %0 = alloca %B, align 4

  %1 = getelementptr %B, ptr %0, i32 1
  %2 = getelementptr inbounds %B, ptr %0, i32 1

  %3 = getelementptr %B, ptr %0, i32 0, i32 0
  %4 = getelementptr inbounds %B, ptr %0, i32 0, i32 0

  %5 = getelementptr %B, ptr %0, i32 2, i32 1
  %6 = getelementptr inbounds %B, ptr %0, i32 2, i32 1

  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }

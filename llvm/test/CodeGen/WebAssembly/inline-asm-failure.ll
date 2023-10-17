; RUN: not llc -O0 --mtriple=wasm32 -filetype=obj \
; RUN:     -o /dev/null 2>&1 <%s | FileCheck %s
source_filename = "rust-issue-111471.c"
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

@__main_void = hidden alias i32 (), ptr @main

; Function Attrs: noinline nounwind optnone
define hidden void @get_global() #0 {
entry:
  call void asm sideeffect "global.get 0", ""() #1, !srcloc !0
  ret void
}

; Function Attrs: noinline nounwind optnone
define hidden i32 @main() #0 {
entry:
  ret i32 0
}

attributes #0 = { noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+mutable-globals,+sign-ext" }
attributes #1 = { nounwind }

!0 = !{i64 32}
; CHECK: <unknown>:0: error: Wasm globals should only be accessed symbolically!

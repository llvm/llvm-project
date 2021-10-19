; RUN: llc --relocation-model=static < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@global = external dso_local global i32
declare dso_local void @func()

define i32* @global_addr() #0 {
  ; CHECK-LABEL: global_addr:
  ; CHECK: movq global@GOTPCREL(%rip), %rax
  ; CHECK: retq

  ret i32* @global
}

define i32 @global_load() #0 {
  ; CHECK-LABEL: global_load:
  ; CHECK: movq global@GOTPCREL(%rip), [[REG:%r[0-9a-z]+]]
  ; CHECK: movl ([[REG]]), %eax
  ; CHECK: retq

  %load = load i32, i32* @global
  ret i32 %load
}

define void @global_store() #0 {
  ; CHECK-LABEL: global_store:
  ; CHECK: movq global@GOTPCREL(%rip), [[REG:%r[0-9a-z]+]]
  ; CHECK: movl $0, ([[REG]])
  ; CHECK: retq

  store i32 0, i32* @global
  ret void
}

define void ()* @func_addr() #0 {
  ; CHECK-LABEL: func_addr:
  ; CHECK: movl $func, %eax
  ; CHECK: retq

  ret void ()* @func
}

attributes #0 = { "target-features"="+tagged-globals" }

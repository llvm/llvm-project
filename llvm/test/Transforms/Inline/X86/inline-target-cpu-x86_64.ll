; RUN: opt < %s -mtriple=x86_64-unknown-unknown -S -passes=inline | FileCheck %s

define i32 @func_target_cpu_base() #0 {
  ret i32 0
}

; CHECK-LABEL: @target_cpu_k8_call_target_cpu_base(
; CHECK-NEXT: ret i32 0
define i32 @target_cpu_k8_call_target_cpu_base() #1 {
  %call = call i32 @func_target_cpu_base()
  ret i32 %call
}

; CHECK-LABEL: @target_cpu_target_nehalem_call_target_cpu_base(
; CHECK-NEXT: ret i32 0
define i32 @target_cpu_target_nehalem_call_target_cpu_base() #2 {
  %call = call i32 @func_target_cpu_base()
  ret i32 %call
}

; CHECK-LABEL: @target_cpu_target_goldmont_call_target_cpu_base(
; CHECK-NEXT: ret i32 0
define i32 @target_cpu_target_goldmont_call_target_cpu_base() #3 {
  %call = call i32 @func_target_cpu_base()
  ret i32 %call
}

define i32 @func_target_cpu_nocona() #4 {
  ret i32 0
}

; CHECK-LABEL: @target_cpu_target_base_call_target_cpu_nocona(
; CHECK-NEXT: ret i32 0
define i32 @target_cpu_target_base_call_target_cpu_nocona() #0 {
  %call = call i32 @func_target_cpu_nocona()
  ret i32 %call
}

attributes #0 = { nounwind "target-cpu"="x86-64" }
attributes #1 = { nounwind "target-cpu"="k8" }
attributes #2 = { nounwind "target-cpu"="nehalem" }
attributes #3 = { nounwind "target-cpu"="goldmont" }
attributes #4 = { nounwind "target-cpu"="nocona" "target-features"="-sse3" }

; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -inline < %s | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes='cgscc(inline)' < %s | FileCheck %s

; CHECK-LABEL: @func_no_target_cpu(
define i32 @func_no_target_cpu() #0 {
  ret i32 0
}

; CHECK-LABEL: @target_cpu_call_no_target_cpu(
; CHECK-NEXT: ret i32 0
define i32 @target_cpu_call_no_target_cpu() #1 {
  %call = call i32 @func_no_target_cpu()
  ret i32 %call
}

; CHECK-LABEL: @target_cpu_target_features_call_no_target_cpu(
; CHECK-NEXT: ret i32 0
define i32 @target_cpu_target_features_call_no_target_cpu() {
  %call = call i32 @func_no_target_cpu()
  ret i32 %call
}

; Make sure gfx9 can call unspecified functions because of movrel
; feature change.
; CHECK-LABEL: @gfx9_target_features_call_no_target_cpu(
; CHECK-NEXT: ret i32 0
define i32 @gfx9_target_features_call_no_target_cpu() #2 {
  %call = call i32 @func_no_target_cpu()
  ret i32 %call
}

define i32 @func_no_halfrate64ops() #3 {
  ret i32 0
}

define i32 @func_with_halfrate64ops() #4 {
  ret i32 0
}

; CHECK-LABEL: @call_func_without_halfrate64ops(
; CHECK-NEXT: ret i32 0
define i32 @call_func_without_halfrate64ops() #4 {
  %call = call i32 @func_no_halfrate64ops()
  ret i32 %call
}

; CHECK-LABEL: @call_func_with_halfrate64ops(
; CHECK-NEXT: ret i32 0
define i32 @call_func_with_halfrate64ops() #3 {
  %call = call i32 @func_with_halfrate64ops()
  ret i32 %call
}

define i32 @func_no_loadstoreopt() #5 {
  ret i32 0
}

define i32 @func_with_loadstoreopt() #6 {
  ret i32 0
}

; CHECK-LABEL: @call_func_without_loadstoreopt(
; CHECK-NEXT: ret i32 0
define i32 @call_func_without_loadstoreopt() #6 {
  %call = call i32 @func_no_loadstoreopt()
  ret i32 %call
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "target-cpu"="fiji" }
attributes #2 = { nounwind "target-cpu"="gfx900" }
attributes #3 = { nounwind "target-features"="-half-rate-64-ops" }
attributes #4 = { nounwind "target-features"="+half-rate-64-ops" }
attributes #5 = { nounwind "target-features"="-load-store-opt" }
attributes #6 = { nounwind "target-features"="+load-store-opt" }

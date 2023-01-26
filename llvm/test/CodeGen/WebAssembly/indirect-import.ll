; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -verify-machineinstrs -fast-isel | FileCheck %s

; ModuleID = 'test/dot_s/indirect-import.c'
source_filename = "test/dot_s/indirect-import.c"
target triple = "wasm32"

; CHECK: .functype extern_fd (f64) -> (f32)
; CHECK: .functype extern_vj (i64) -> ()
; CHECK: .functype extern_v () -> ()
; CHECK: .functype extern_ijidf  (i64, i32, f64, f32) -> (i32)
; CHECK: .functype extern_struct (i32) -> ()
; CHECK: .functype extern_sret (i32) -> ()
; CHECK: .functype extern_i128ret (i32, i64) -> ()

%struct.big = type { float, double, i32 }

; Function Attrs: nounwind
; CHECK-LABEL: bar:
define hidden i32 @bar() #0 {
entry:
  %fd = alloca ptr, align 4
  %vj = alloca ptr, align 4
  %v = alloca ptr, align 4
  %ijidf = alloca ptr, align 4
  %vs = alloca ptr, align 4
  %s = alloca ptr, align 4
  %i128ret = alloca ptr, align 8

; CHECK-DAG: i32.const       {{.+}}=, extern_fd
; CHECK-DAG: i32.const       {{.+}}=, extern_vj
  store ptr @extern_fd, ptr %fd, align 4
  store ptr @extern_vj, ptr %vj, align 4
  %0 = load ptr, ptr %vj, align 4
  call void %0(i64 1)

; CHECK: i32.const       {{.+}}=, extern_v
  store ptr @extern_v, ptr %v, align 4
  %1 = load ptr, ptr %v, align 4
  call void %1()

; CHECK: i32.const       {{.+}}=, extern_ijidf
  store ptr @extern_ijidf, ptr %ijidf, align 4
  %2 = load ptr, ptr %ijidf, align 4
  %call = call i32 %2(i64 1, i32 2, double 3.000000e+00, float 4.000000e+00)

; CHECK-DAG: i32.const       {{.+}}=, extern_struct
; CHECK-DAG: i32.const       {{.+}}=, extern_sret
  store ptr @extern_struct, ptr %vs, align 4
  store ptr @extern_sret, ptr %s, align 4
  %3 = load ptr, ptr %fd, align 4
  %4 = ptrtoint ptr %3 to i32

; CHECK: i32.const       {{.+}}=, extern_i128ret
  store ptr @extern_i128ret, ptr %i128ret, align 8
  %5 = load ptr, ptr %i128ret, align 8
  %6 = call i128 %5(i64 1)

  ret i32 %4
}

declare float @extern_fd(double) #1

declare void @extern_vj(i64) #1

declare void @extern_v() #1

declare i32 @extern_ijidf(i64, i32, double, float) #1

declare void @extern_struct(ptr byval(%struct.big) align 8) #1

declare void @extern_sret(ptr sret(%struct.big)) #1

declare i128 @extern_i128ret(i64) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

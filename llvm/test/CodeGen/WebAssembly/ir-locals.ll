; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false | FileCheck %s

%i32_cell = type i32 addrspace(1)*
%i64_cell = type i64 addrspace(1)*
%f32_cell = type float addrspace(1)*
%f64_cell = type double addrspace(1)*

; We have a set of tests in which we set a local and then reload the
; local.  If the load immediately follows the set, the DAG combiner will
; infer that the reloaded value is the same value that was set, which
; isn't what we want to test.  To inhibit this optimization, we include
; an opaque call between the store and the load.
declare void @inhibit_store_to_load_forwarding()

define i32 @ir_local_i32(i32 %arg) {
 ; CHECK-LABEL: ir_local_i32:
 ; CHECK-NEXT: .functype ir_local_i32 (i32) -> (i32)
 %retval = alloca i32, addrspace(1)
 ; CHECK-NEXT: .local i32
 store i32 %arg, %i32_cell %retval
 ; CHECK-NEXT: local.get 0
 ; CHECK-NEXT: local.set 1
 call void @inhibit_store_to_load_forwarding()
 ; CHECK-NEXT: call inhibit_store_to_load_forwarding
 %reloaded = load i32, %i32_cell %retval
 ; CHECK-NEXT: local.get 1
 ret i32 %reloaded
 ; CHECK-NEXT: end_function
}

define i64 @ir_local_i64(i64 %arg) {
 ; CHECK-LABEL: ir_local_i64:
 ; CHECK-NEXT: .functype ir_local_i64 (i64) -> (i64)
 %retval = alloca i64, addrspace(1)
 ; CHECK-NEXT: .local i64
 store i64 %arg, %i64_cell %retval
 ; CHECK-NEXT: local.get 0
 ; CHECK-NEXT: local.set 1
 call void @inhibit_store_to_load_forwarding()
 ; CHECK-NEXT: call inhibit_store_to_load_forwarding
 %reloaded = load i64, %i64_cell %retval
 ; See note in ir_local_i32.
 ; CHECK-NEXT: local.get 1
 ret i64 %reloaded
 ; CHECK-NEXT: end_function
}

define float @ir_local_f32(float %arg) {
 ; CHECK-LABEL: ir_local_f32:
 ; CHECK-NEXT: .functype ir_local_f32 (f32) -> (f32)
 %retval = alloca float, addrspace(1)
 ; CHECK-NEXT: .local f32
 store float %arg, %f32_cell %retval
 ; CHECK-NEXT: local.get 0
 ; CHECK-NEXT: local.set 1
 call void @inhibit_store_to_load_forwarding()
 ; CHECK-NEXT: call inhibit_store_to_load_forwarding
 %reloaded = load float, %f32_cell %retval
 ; CHECK-NEXT: local.get 1
 ; CHECK-NEXT: end_function
 ret float %reloaded
}

define double @ir_local_f64(double %arg) {
 ; CHECK-LABEL: ir_local_f64:
 ; CHECK-NEXT: .functype ir_local_f64 (f64) -> (f64)
 %retval = alloca double, addrspace(1)
 ; CHECK-NEXT: .local f64
 store double %arg, %f64_cell %retval
 ; CHECK-NEXT: local.get 0
 ; CHECK-NEXT: local.set 1
 call void @inhibit_store_to_load_forwarding()
 ; CHECK-NEXT: call inhibit_store_to_load_forwarding
 %reloaded = load double, %f64_cell %retval
 ; CHECK-NEXT: local.get 1
 ; CHECK-NEXT: end_function
 ret double %reloaded
}

define void @ir_unreferenced_local() {
 ; CHECK-LABEL: ir_unreferenced_local:
 ; CHECK-NEXT: .functype ir_unreferenced_local () -> ()
 %unused = alloca i32, addrspace(1)
 ; CHECK-NEXT: .local i32
 ret void
 ; CHECK-NEXT: end_function
}

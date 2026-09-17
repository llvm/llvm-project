; RUN: mlir-translate -import-llvm %s -split-input-file | FileCheck %s

declare i64 @llvm.aarch64.ldxr.p0(ptr)

define dso_local void @t0(ptr %a) {
  %x = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i8) %a)
  ret void
}

; CHECK-LABEL: llvm.func @llvm.aarch64.ldxr.p0(!llvm.ptr)
; CHECK-LABEL: llvm.func @t0
; CHECK:   llvm.call_intrinsic "llvm.aarch64.ldxr.p0"({{.*}}) : (!llvm.ptr {llvm.elementtype = i8}) -> i64
; CHECK:   llvm.return

; // -----

declare <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8>, <8 x i8>)

define dso_local <8 x i8> @t1(<8 x i8> %lhs, <8 x i8> %rhs) {
  %r = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
  ret <8 x i8> %r
}

; CHECK: llvm.func @t1(%[[A0:.*]]: vector<8xi8>, %[[A1:.*]]: vector<8xi8>) -> vector<8xi8> {{.*}}
; CHECK:   %[[R:.*]] = llvm.call_intrinsic "llvm.aarch64.neon.uabd.v8i8"(%[[A0]], %[[A1]]) : (vector<8xi8>, vector<8xi8>) -> vector<8xi8>
; CHECK:   llvm.return %[[R]] : vector<8xi8>

; // -----

declare void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8>, <8 x i8>, ptr)

define dso_local void @t2(<8 x i8> %lhs, <8 x i8> %rhs, ptr %a) {
  call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> %lhs, <8 x i8> %rhs, ptr %a)
  ret void
}

; CHECK: llvm.func @t2(%[[A0:.*]]: vector<8xi8>, %[[A1:.*]]: vector<8xi8>, %[[A2:.*]]: !llvm.ptr) {{.*}}
; CHECK:   llvm.call_intrinsic "llvm.aarch64.neon.st2.v8i8.p0"(%[[A0]], %[[A1]], %[[A2]]) : (vector<8xi8>, vector<8xi8>, !llvm.ptr) -> ()
; CHECK:   llvm.return

; // -----

declare void @llvm.gcroot(ptr %arg1, ptr %arg2)
define void @gctest() gc "example" {
  %arg1 = alloca ptr
  call void @llvm.gcroot(ptr %arg1, ptr null)
  ret void
}

; CHECK-LABEL: @gctest
; CHECK: llvm.call_intrinsic "llvm.gcroot"({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()

; // -----

; Test we get the supported version, not the unregistered one.

declare i32 @llvm.lround.i32.f32(float)

; CHECK-LABEL: llvm.func @lround_test
define void @lround_test(float %0, double %1) {
  ; CHECK-NOT: llvm.call_intrinsic "llvm.lround
  ; CHECK: llvm.intr.lround(%{{.*}}) : (f32) -> i32
  %3 = call i32 @llvm.lround.i32.f32(float %0)
  ret void
}

; // -----

declare i32 @llvm.riscv.sha256sig0(i32)

; CHECK-LABEL: test_intrin_arg_attr
define signext i32 @test_intrin_arg_attr(i32 signext %a) nounwind {
    ; CHECK: llvm.call_intrinsic "llvm.riscv.sha256sig0"({{.*}}) : (i32 {llvm.signext}) -> i32
    %val = call i32 @llvm.riscv.sha256sig0(i32 signext %a)
    ret i32 %val
}

; // -----

; Rounding FP intrinsics with no dedicated MLIR op should fall back to
; `llvm.call_intrinsic`, and their `metadata !"..."` operands should be
; imported as `llvm.mlir.metadata_as_value` ops wrapping the corresponding
; `#llvm.md_string` attribute. `llvm.fptrunc.round` is used here because it
; takes an MDString rounding-mode operand and has no specialized MLIR op.

declare float @llvm.fptrunc.round.f32.f64(double, metadata)

; CHECK-LABEL: llvm.func @fptrunc_round
define float @fptrunc_round(double %a) {
  ; CHECK: %[[RM:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
  ; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.fptrunc.round.f32.f64"(%{{.*}}, %[[RM]]) : (f64, !llvm.metadata) -> f32
  %r = call float @llvm.fptrunc.round.f32.f64(double %a, metadata !"round.tonearest")
  ret float %r
}

; // -----

; Importer should also handle MDNode metadata operands such as the
; `!{!"register_name"}` form used by `llvm.read_register`.

declare i32 @llvm.read_register.i32(metadata)

; CHECK-LABEL: llvm.func @read_named_register
define i32 @read_named_register() {
  ; CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_node<#llvm.md_string<"sp">>
  ; CHECK: %[[R0:.*]] = llvm.call_intrinsic "llvm.read_register.i32"(%[[MD]]) : (!llvm.metadata) -> i32
  %r0 = call i32 @llvm.read_register.i32(metadata !0)
  ; CHECK-NOT: llvm.mlir.metadata_as_value
  ; CHECK: %[[R1:.*]] = llvm.call_intrinsic "llvm.read_register.i32"(%[[MD]]) : (!llvm.metadata) -> i32
  %r1 = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %r1
}

!0 = !{!"sp"}

; // -----

declare i32 @llvm.read_register.i32(metadata)

@global = global i32 0

; CHECK: llvm.mlir.global external @[[$GLOBAL:global]]
; CHECK-LABEL: llvm.func @read_global_metadata
define i32 @read_global_metadata() {
  ; CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_global_value<@[[$GLOBAL]]>
  ; CHECK: llvm.call_intrinsic "llvm.read_register.i32"(%[[MD]]) : (!llvm.metadata) -> i32
  %r = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %r
}

!0 = !{ptr @global}

; // -----

declare i32 @llvm.read_register.i32(metadata)

@0 = global i32 0

; CHECK: llvm.mlir.global external @[[$NAMELESS_GLOBAL:mlir\.llvm\.nameless_global_[0-9]+]]
; CHECK-LABEL: llvm.func @read_nameless_global_metadata
define i32 @read_nameless_global_metadata() {
  ; CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_global_value<@[[$NAMELESS_GLOBAL]]>
  ; CHECK: llvm.call_intrinsic "llvm.read_register.i32"(%[[MD]]) : (!llvm.metadata) -> i32
  %r = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %r
}

!0 = !{ptr @0}

; // -----

declare i32 @llvm.read_register.i32(metadata)
declare void @callee()

; CHECK: llvm.func @[[$CALLEE:callee]]()
; CHECK-LABEL: llvm.func @read_function_metadata
define i32 @read_function_metadata() {
  ; CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_global_value<@[[$CALLEE]]>
  ; CHECK: llvm.call_intrinsic "llvm.read_register.i32"(%[[MD]]) : (!llvm.metadata) -> i32
  %r = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %r
}

!0 = !{ptr @callee}

; // -----

declare i32 @llvm.read_register.i32(metadata)

define void @alias_target() {
  ret void
}
@alias = alias void (), ptr @alias_target

; CHECK: llvm.mlir.alias external @[[$ALIAS:alias]]
; CHECK-LABEL: llvm.func @read_alias_metadata
define i32 @read_alias_metadata() {
  ; CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_global_value<@[[$ALIAS]]>
  ; CHECK: llvm.call_intrinsic "llvm.read_register.i32"(%[[MD]]) : (!llvm.metadata) -> i32
  %r = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %r
}

!0 = !{ptr @alias}

; // -----

declare i32 @llvm.read_register.i32(metadata)

@ifunc = ifunc void (), ptr @ifunc_resolver
define ptr @ifunc_resolver() {
  ret ptr @ifunc_target
}
define void @ifunc_target() {
  ret void
}

; CHECK: llvm.mlir.ifunc external @[[$IFUNC:ifunc]]
; CHECK-LABEL: llvm.func @read_ifunc_metadata
define i32 @read_ifunc_metadata() {
  ; CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_global_value<@[[$IFUNC]]>
  ; CHECK: llvm.call_intrinsic "llvm.read_register.i32"(%[[MD]]) : (!llvm.metadata) -> i32
  %r = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %r
}

!0 = !{ptr @ifunc}

; // -----

; A null pointer constant metadata operand must be preserved.

declare i32 @llvm.read_register.i32(metadata)

; CHECK-LABEL: llvm.func @read_null_metadata
define i32 @read_null_metadata() {
  ; CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_null<0>
  ; CHECK: llvm.call_intrinsic "llvm.read_register.i32"(%[[MD]]) : (!llvm.metadata) -> i32
  %r = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %r
}

!0 = !{ptr null}

; // -----

; The address space of a null pointer constant must be preserved.

declare i32 @llvm.read_register.i32(metadata)

; CHECK-LABEL: llvm.func @read_null_addrspace_metadata
define i32 @read_null_addrspace_metadata() {
  ; CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_null<1>
  ; CHECK: llvm.call_intrinsic "llvm.read_register.i32"(%[[MD]]) : (!llvm.metadata) -> i32
  %r = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %r
}

!0 = !{ptr addrspace(1) null}

; // -----

; An addrspacecast constant expression metadata operand must be preserved.

declare i32 @llvm.read_register.i32(metadata)

@addrspace_global = addrspace(1) global i32 0

; CHECK: llvm.mlir.global external @[[$GLOBAL:addrspace_global]]
; CHECK-LABEL: llvm.func @read_addrspacecast_metadata
define i32 @read_addrspacecast_metadata() {
  ; CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_addrspacecast<#llvm.md_global_value<@[[$GLOBAL]]>, 0>
  ; CHECK: llvm.call_intrinsic "llvm.read_register.i32"(%[[MD]]) : (!llvm.metadata) -> i32
  %r = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %r
}

!0 = !{ptr addrspacecast (ptr addrspace(1) @addrspace_global to ptr)}

; // -----

; Pointer constants nested inside a multi-operand metadata node.

declare i32 @llvm.read_register.i32(metadata)

@nested_global = addrspace(1) global i32 0

; CHECK: llvm.mlir.global external @[[$GLOBAL:nested_global]]
; CHECK-LABEL: llvm.func @read_pointer_constants_in_node
define i32 @read_pointer_constants_in_node() {
  ; CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_node<#llvm.md_null<0>, #llvm.md_addrspacecast<#llvm.md_global_value<@[[$GLOBAL]]>, 0>>
  ; CHECK: llvm.call_intrinsic "llvm.read_register.i32"(%[[MD]]) : (!llvm.metadata) -> i32
  %r = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %r
}

!0 = !{ptr null, ptr addrspacecast (ptr addrspace(1) @nested_global to ptr)}

// RUN: mlir-opt %s -inline -split-input-file | FileCheck %s

#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "foo">
#alias_scope = #llvm.alias_scope<id = distinct[1]<>, domain = #alias_scope_domain, description = "foo load">
#alias_scope1 = #llvm.alias_scope<id = distinct[2]<>, domain = #alias_scope_domain, description = "foo store">

// CHECK-DAG: #[[FOO_DOMAIN:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$FOO_LOAD:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[FOO_DOMAIN]], description = {{.*}}>
// CHECK-DAG: #[[$FOO_STORE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[FOO_DOMAIN]], description = {{.*}}>

// CHECK-DAG: #[[BAR_DOMAIN:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$BAR_LOAD:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[BAR_DOMAIN]], description = {{.*}}>
// CHECK-DAG: #[[$BAR_STORE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[BAR_DOMAIN]], description = {{.*}}>

// CHECK-LABEL: llvm.func @foo
// CHECK: llvm.intr.experimental.noalias.scope.decl #[[$FOO_LOAD]]
// CHECK: llvm.load
// CHECK-SAME: alias_scopes = [#[[$FOO_LOAD]]]
// CHECK-SAME: noalias_scopes = [#[[$FOO_STORE]]]
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$FOO_STORE]]]
// CHECK-SAME: noalias_scopes = [#[[$FOO_LOAD]]]
llvm.func @foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %0 = llvm.mlir.constant(5 : i64) : i64
  llvm.intr.experimental.noalias.scope.decl #alias_scope
  %2 = llvm.load %arg1 {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1]} : !llvm.ptr -> f32
  "test.one_region_op"() ({
    %3 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %3 {alias_scopes = [#alias_scope1], alignment = 4 : i64, noalias_scopes = [#alias_scope]} : f32, !llvm.ptr
    "test.terminator"() : () -> ()
  }) : () -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @clone_alias_scopes
// CHECK: llvm.intr.experimental.noalias.scope.decl #[[$BAR_LOAD]]
// CHECK: llvm.load
// CHECK-SAME: alias_scopes = [#[[$BAR_LOAD]]]
// CHECK-SAME: noalias_scopes = [#[[$BAR_STORE]]]
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$BAR_STORE]]]
// CHECK-SAME: noalias_scopes = [#[[$BAR_LOAD]]]
llvm.func @clone_alias_scopes(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  llvm.call @foo(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "hello2">
#alias_scope_domain1 = #llvm.alias_scope_domain<id = distinct[1]<>, description = "hello">
#alias_scope = #llvm.alias_scope<id = distinct[2]<>, domain = #alias_scope_domain, description = "hello2: %a">
#alias_scope1 = #llvm.alias_scope<id = distinct[3]<>, domain = #alias_scope_domain, description = "hello2: %b">
#alias_scope2 = #llvm.alias_scope<id = distinct[4]<>, domain = #alias_scope_domain1, description = "hello: %a">

// CHECK-DAG: #[[WITH_DOMAIN:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$WITH_DOMAIN_SCOPE1:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[WITH_DOMAIN]], description = {{.*}}>
// CHECK-DAG: #[[$WITH_DOMAIN_SCOPE2:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[WITH_DOMAIN]], description = {{.*}}>

// CHECK-DAG: #[[CALL_DOMAIN:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$CALL_DOMAIN_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[CALL_DOMAIN]], description = {{.*}}>

// CHECK-DAG: #[[WITH_DOMAIN_NO_ALIAS:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$WITH_DOMAIN_NO_ALIAS_SCOPE1:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[WITH_DOMAIN_NO_ALIAS]], description = {{.*}}>
// CHECK-DAG: #[[$WITH_DOMAIN_NO_ALIAS_SCOPE2:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[WITH_DOMAIN_NO_ALIAS]], description = {{.*}}>

// CHECK-DAG: #[[WITH_DOMAIN_ALIAS:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$WITH_DOMAIN_ALIAS_SCOPE1:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[WITH_DOMAIN_ALIAS]], description = {{.*}}>
// CHECK-DAG: #[[$WITH_DOMAIN_ALIAS_SCOPE2:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[WITH_DOMAIN_ALIAS]], description = {{.*}}>

// CHECK-LABEL: llvm.func @callee_with_metadata(
// CHECK: llvm.load
// CHECK-SAME: noalias_scopes = [#[[$WITH_DOMAIN_SCOPE1]], #[[$WITH_DOMAIN_SCOPE2]]]
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$WITH_DOMAIN_SCOPE1]]]
// CHECK-SAME: noalias_scopes = [#[[$WITH_DOMAIN_SCOPE2]]]
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$WITH_DOMAIN_SCOPE2]]]
// CHECK-SAME: noalias_scopes = [#[[$WITH_DOMAIN_SCOPE1]]]
// CHECK: llvm.load
// CHECK-NOT: {{(no)?}}alias_scopes =
// CHECK: llvm.store
// CHECK-NOT: {{(no)?}}alias_scopes =
llvm.func @callee_with_metadata(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  %0 = llvm.mlir.constant(5 : i64) : i64
  %1 = llvm.mlir.constant(8 : i64) : i64
  %2 = llvm.mlir.constant(7 : i64) : i64
  %3 = llvm.load %arg2 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1]} : !llvm.ptr -> f32
  %4 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %3, %4 {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1]} : f32, !llvm.ptr
  %5 = llvm.getelementptr inbounds %arg1[%1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %3, %5 {alias_scopes = [#alias_scope1], alignment = 4 : i64, noalias_scopes = [#alias_scope]} : f32, !llvm.ptr
  "test.one_region_op"() ({
    %6 = llvm.load %arg2 {alignment = 4 : i64} : !llvm.ptr -> f32
    %7 = llvm.getelementptr inbounds %arg0[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %6, %7 {alignment = 4 : i64} : f32, !llvm.ptr
    "test.terminator"() : () -> ()
  }) : () -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @callee_without_metadata(
// CHECK-NOT: {{(no)?}}alias_scopes =

llvm.func @callee_without_metadata(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  %0 = llvm.mlir.constant(5 : i64) : i64
  %1 = llvm.mlir.constant(8 : i64) : i64
  %2 = llvm.mlir.constant(7 : i64) : i64
  %3 = llvm.load %arg2 {alignment = 4 : i64} : !llvm.ptr -> f32
  %4 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %3, %4 {alignment = 4 : i64} : f32, !llvm.ptr
  %5 = llvm.getelementptr inbounds %arg1[%1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %3, %5 {alignment = 4 : i64} : f32, !llvm.ptr
  "test.one_region_op"() ({
    %6 = llvm.load %arg2 {alignment = 4 : i64} : !llvm.ptr -> f32
    %7 = llvm.getelementptr inbounds %arg0[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %6, %7 {alignment = 4 : i64} : f32, !llvm.ptr
    "test.terminator"() : () -> ()
  }) : () -> ()

  llvm.return
}

// CHECK-LABEL: llvm.func @caller(
// CHECK: llvm.load
// CHECK-SAME: alias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK-NOT: noalias_scopes

// Inlining @callee_with_metadata with noalias_scopes.

// CHECK: llvm.load
// CHECK-SAME: noalias_scopes = [#[[$WITH_DOMAIN_NO_ALIAS_SCOPE1]], #[[$WITH_DOMAIN_NO_ALIAS_SCOPE2]], #[[$CALL_DOMAIN_SCOPE]]]
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$WITH_DOMAIN_NO_ALIAS_SCOPE1]]]
// CHECK-SAME: noalias_scopes = [#[[$WITH_DOMAIN_NO_ALIAS_SCOPE2]], #[[$CALL_DOMAIN_SCOPE]]]
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$WITH_DOMAIN_NO_ALIAS_SCOPE2]]]
// CHECK-SAME: noalias_scopes = [#[[$WITH_DOMAIN_NO_ALIAS_SCOPE1]], #[[$CALL_DOMAIN_SCOPE]]]
// CHECK: llvm.load
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK: llvm.store
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]

// Inlining @callee_with_metadata with alias_scopes.

// CHECK: llvm.load
// CHECK-SAME: alias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK-SAME: noalias_scopes = [#[[$WITH_DOMAIN_ALIAS_SCOPE1]], #[[$WITH_DOMAIN_ALIAS_SCOPE2]]]
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$WITH_DOMAIN_ALIAS_SCOPE1]], #[[$CALL_DOMAIN_SCOPE]]]
// CHECK-SAME: noalias_scopes = [#[[$WITH_DOMAIN_ALIAS_SCOPE2]]]
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$WITH_DOMAIN_ALIAS_SCOPE2]], #[[$CALL_DOMAIN_SCOPE]]]
// CHECK-SAME: noalias_scopes = [#[[$WITH_DOMAIN_ALIAS_SCOPE1]]]
// CHECK: llvm.load
// CHECK-SAME: alias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK-NOT: noalias_scopes
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK-NOT: noalias_scopes

// Inlining @callee_without_metadata with noalias_scopes.

// CHECK: llvm.load
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK: llvm.store
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK: llvm.store
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK: llvm.load
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK: llvm.store
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]

// Inlining @callee_without_metadata with alias_scopes.

// CHECK: llvm.load
// CHECK-SAME: alias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK-NOT: noalias_scopes
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK-NOT: noalias_scopes
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK-NOT: noalias_scopes
// CHECK: llvm.load
// CHECK-SAME: alias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK-NOT: noalias_scopes
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$CALL_DOMAIN_SCOPE]]]
// CHECK-NOT: noalias_scopes

llvm.func @caller(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  %0 = llvm.load %arg2 {alias_scopes = [#alias_scope2], alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  llvm.call @callee_with_metadata(%arg0, %arg1, %0) {noalias_scopes = [#alias_scope2]} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.call @callee_with_metadata(%arg1, %arg1, %arg0) {alias_scopes = [#alias_scope2]} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.call @callee_without_metadata(%arg0, %arg1, %0) {noalias_scopes = [#alias_scope2]} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.call @callee_without_metadata(%arg1, %arg1, %arg0) {alias_scopes = [#alias_scope2]} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

// CHECK-DAG: #[[DOMAIN:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$ARG0_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>
// CHECK-DAG: #[[$ARG1_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>

llvm.func @foo(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}) {
  %0 = llvm.mlir.constant(5 : i64) : i64
  %1 = llvm.load %arg1 {alignment = 4 : i64} : !llvm.ptr -> f32
  %2 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %1, %2 {alignment = 4 : i64} : f32, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: llvm.func @bar
// CHECK: llvm.load
// CHECK-SAME: alias_scopes = [#[[$ARG1_SCOPE]]]
// CHECK-SAME: noalias_scopes = [#[[$ARG0_SCOPE]]]
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$ARG0_SCOPE]]]
// CHECK-SAME: noalias_scopes = [#[[$ARG1_SCOPE]]]
llvm.func @bar(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  llvm.call @foo(%arg0, %arg2) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

// CHECK-DAG: #[[DOMAIN:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$ARG0_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>
// CHECK-DAG: #[[$ARG1_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>

llvm.func @might_return_arg_derived(!llvm.ptr) -> !llvm.ptr

llvm.func @foo(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}) {
  %0 = llvm.mlir.constant(5 : i64) : i32
  %1 = llvm.call @might_return_arg_derived(%arg0) : (!llvm.ptr) -> !llvm.ptr
  llvm.store %0, %1 : i32, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: llvm.func @bar
// CHECK: llvm.call
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$ARG1_SCOPE]]]
// CHECK: llvm.store
// CHECK-NOT: alias_scopes
// CHECK-NOT: noalias_scopes
llvm.func @bar(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  llvm.call @foo(%arg0, %arg2) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

// CHECK-DAG: #[[DOMAIN:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$ARG0_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>
// CHECK-DAG: #[[$ARG1_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>

llvm.func @random() -> i1

llvm.func @block_arg(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}) {
  %0 = llvm.mlir.constant(5 : i64) : i32
  %1 = llvm.call @random() : () -> i1
  llvm.cond_br %1, ^bb0(%arg0 : !llvm.ptr), ^bb0(%arg1 : !llvm.ptr)

^bb0(%arg2: !llvm.ptr):
  llvm.store %0, %arg2 : i32, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: llvm.func @bar
// CHECK: llvm.call
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$ARG0_SCOPE]], #[[$ARG1_SCOPE]]]
// CHECK: llvm.store
// CHECK: alias_scopes = [#[[$ARG0_SCOPE]], #[[$ARG1_SCOPE]]]
llvm.func @bar(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  llvm.call @block_arg(%arg0, %arg2) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

// CHECK-DAG: #[[DOMAIN:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$ARG0_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>
// CHECK-DAG: #[[$ARG1_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>

llvm.func @random() -> i1

llvm.func @block_arg(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}) {
  %0 = llvm.mlir.constant(5 : i64) : i32
  %1 = llvm.mlir.constant(1 : i64) : i64
  %2 = llvm.alloca %1 x i32 : (i64) -> !llvm.ptr
  %3 = llvm.call @random() : () -> i1
  llvm.cond_br %3, ^bb0(%arg0 : !llvm.ptr), ^bb0(%2 : !llvm.ptr)

^bb0(%arg2: !llvm.ptr):
  llvm.store %0, %arg2 : i32, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: llvm.func @bar
// CHECK: llvm.call
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$ARG0_SCOPE]], #[[$ARG1_SCOPE]]]
// CHECK: llvm.store
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$ARG1_SCOPE]]]
llvm.func @bar(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  llvm.call @block_arg(%arg0, %arg2) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

// CHECK-DAG: #[[DOMAIN:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$ARG0_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>
// CHECK-DAG: #[[$ARG1_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>

llvm.func @unknown() -> !llvm.ptr
llvm.func @random() -> i1

llvm.func @unknown_object(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}) {
  %0 = llvm.mlir.constant(5 : i64) : i32
  %1 = llvm.call @random() : () -> i1
  %2 = llvm.call @unknown() : () -> !llvm.ptr
  llvm.cond_br %1, ^bb0(%arg0 : !llvm.ptr), ^bb0(%2 : !llvm.ptr)

^bb0(%arg2: !llvm.ptr):
  llvm.store %0, %arg2 : i32, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: llvm.func @bar
// CHECK: llvm.call
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$ARG0_SCOPE]], #[[$ARG1_SCOPE]]]
// CHECK: llvm.call
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$ARG0_SCOPE]], #[[$ARG1_SCOPE]]]
// CHECK: llvm.store
// CHECK-NOT: alias_scopes
// CHECK-NOT: noalias_scopes
llvm.func @bar(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  llvm.call @unknown_object(%arg0, %arg2) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

// CHECK-DAG: #[[DOMAIN:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$ARG0_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>
// CHECK-DAG: #[[$ARG1_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>

llvm.func @supported_operations(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}) {
  %0 = llvm.mlir.constant(5 : i64) : i32
  llvm.store %0, %arg1 : i32, !llvm.ptr
  %1 = llvm.load %arg1 : !llvm.ptr -> i32
  "llvm.intr.memcpy"(%arg0, %arg1, %1) <{ isVolatile = false }> : (!llvm.ptr, !llvm.ptr, i32) -> ()
  "llvm.intr.memmove"(%arg0, %arg1, %1) <{ isVolatile = false }> : (!llvm.ptr, !llvm.ptr, i32) -> ()
  "llvm.intr.memcpy.inline"(%arg0, %arg1) <{ isVolatile = false, len = 4 : i32}> : (!llvm.ptr, !llvm.ptr) -> ()
  %2 = llvm.trunc %0 : i32 to i8
  "llvm.intr.memset"(%arg0, %2, %1) <{ isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %3 = llvm.cmpxchg %arg0, %0, %1 seq_cst seq_cst : !llvm.ptr, i32
  %4 = llvm.atomicrmw add %arg0, %0 seq_cst : !llvm.ptr, i32
  llvm.return
}

// CHECK-LABEL: llvm.func @bar
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$ARG1_SCOPE]]]
// CHECK-SAME: noalias_scopes = [#[[$ARG0_SCOPE]]]
// CHECK: llvm.load
// CHECK-SAME: alias_scopes = [#[[$ARG1_SCOPE]]]
// CHECK-SAME: noalias_scopes = [#[[$ARG0_SCOPE]]]
// CHECK: "llvm.intr.memcpy"
// CHECK-SAME: alias_scopes = [#[[$ARG0_SCOPE]], #[[$ARG1_SCOPE]]]
// CHECK-NOT: noalias_scopes
// CHECK: "llvm.intr.memmove"
// CHECK-SAME: alias_scopes = [#[[$ARG0_SCOPE]], #[[$ARG1_SCOPE]]]
// CHECK-NOT: noalias_scopes
// CHECK: "llvm.intr.memcpy.inline"
// CHECK-SAME: alias_scopes = [#[[$ARG0_SCOPE]], #[[$ARG1_SCOPE]]]
// CHECK-NOT: noalias_scopes
// CHECK: "llvm.intr.memset"
// CHECK-SAME: alias_scopes = [#[[$ARG0_SCOPE]]]
// CHECK-SAME: noalias_scopes = [#[[$ARG1_SCOPE]]]
// CHECK: llvm.cmpxchg
// CHECK-SAME: alias_scopes = [#[[$ARG0_SCOPE]]]
// CHECK-SAME: noalias_scopes = [#[[$ARG1_SCOPE]]]
// CHECK: llvm.atomicrmw
// CHECK-SAME: alias_scopes = [#[[$ARG0_SCOPE]]]
// CHECK-SAME: noalias_scopes = [#[[$ARG1_SCOPE]]]
llvm.func @bar(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  llvm.call @supported_operations(%arg0, %arg2) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

// CHECK-DAG: #[[DOMAIN:.*]] = #llvm.alias_scope_domain<{{.*}}>
// CHECK-DAG: #[[$ARG_SCOPE:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN]]{{(,.*)?}}>

llvm.func @foo(%arg: i32)

llvm.func @region(%arg0: !llvm.ptr {llvm.noalias}) {
  "test.one_region_op"() ({
    %1 = llvm.load %arg0 : !llvm.ptr -> i32
    llvm.call @foo(%1) : (i32) -> ()
    "test.terminator"() : () -> ()
  }) : () -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @noalias_with_region
// CHECK: llvm.load
// CHECK-SAME: alias_scopes = [#[[$ARG_SCOPE]]]
// CHECK: llvm.call
// CHECK-NOT: alias_scopes
// CHECK-SAME: noalias_scopes = [#[[$ARG_SCOPE]]]
llvm.func @noalias_with_region(%arg0: !llvm.ptr) {
  llvm.call @region(%arg0) : (!llvm.ptr) -> ()
  llvm.return
}

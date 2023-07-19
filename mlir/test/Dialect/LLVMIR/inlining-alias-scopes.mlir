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
  %3 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %2, %3 {alias_scopes = [#alias_scope1], alignment = 4 : i64, noalias_scopes = [#alias_scope]} : f32, !llvm.ptr
  llvm.return
}

// CHECK-LABEL: llvm.func @bar
// CHECK: llvm.intr.experimental.noalias.scope.decl #[[$BAR_LOAD]]
// CHECK: llvm.load
// CHECK-SAME: alias_scopes = [#[[$BAR_LOAD]]]
// CHECK-SAME: noalias_scopes = [#[[$BAR_STORE]]]
// CHECK: llvm.store
// CHECK-SAME: alias_scopes = [#[[$BAR_STORE]]]
// CHECK-SAME: noalias_scopes = [#[[$BAR_LOAD]]]
llvm.func @bar(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  llvm.call @foo(%arg0, %arg2) : (!llvm.ptr, !llvm.ptr) -> ()
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
  %6 = llvm.load %arg2 {alignment = 4 : i64} : !llvm.ptr -> f32
  %7 = llvm.getelementptr inbounds %arg0[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %6, %7 {alignment = 4 : i64} : f32, !llvm.ptr
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
  %6 = llvm.load %arg2 {alignment = 4 : i64} : !llvm.ptr -> f32
  %7 = llvm.getelementptr inbounds %arg0[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %6, %7 {alignment = 4 : i64} : f32, !llvm.ptr
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

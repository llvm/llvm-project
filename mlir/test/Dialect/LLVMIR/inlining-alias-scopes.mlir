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

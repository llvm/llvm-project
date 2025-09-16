// RUN: mlir-opt --convert-memref-alias-attributes-to-llvm %s | FileCheck %s

#alias_scope1 = #memref.alias_scope<id = distinct[1]<>, description = "scope">
#alias_scope2 = #memref.alias_scope<id = distinct[2]<>>

// CHECK-DAG: #[[DOMAIN1:.*]] = #llvm.alias_scope_domain<id = {{.*}}, description = "foo">
// CHECK-DAG: #[[DOMAIN2:.*]] = #llvm.alias_scope_domain<id = {{.*}}>
// CHECK-DAG: #[[SCOPE1_1:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN1]], description = "scope">
// CHECK-DAG: #[[SCOPE1_2:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN1]]>
// CHECK-DAG: #[[SCOPE2_1:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN2]], description = "scope">
// CHECK-DAG: #[[SCOPE2_2:.*]] = #llvm.alias_scope<id = {{.*}}, domain = #[[DOMAIN2]]>

// CHECK: func @memref_alias_attributes
func.func @memref_alias_attributes(%arg1 : memref<?xf32>, %arg2 : memref<?xf32>, %arg3: index) -> f32 {
  // CHECK-NOT: alias_domain_scope
  memref.alias_domain_scope "foo" {
  // CHECK:  %[[VAL:.*]] = memref.load %{{.*}}[%{{.*}}] {alias = #memref.aliasing<alias_scopes = [#[[SCOPE1_1]]], noalias = [#[[SCOPE1_2]]]>} : memref<?xf32>
  // CHECK:  memref.store %[[VAL]], %{{.*}}[%{{.*}}] {alias = #memref.aliasing<alias_scopes = [#[[SCOPE1_2]]], noalias = [#[[SCOPE1_1]]]>} : memref<?xf32>
    %val = memref.load %arg1[%arg3] { alias = #memref.aliasing<alias_scopes=[#alias_scope1], noalias=[#alias_scope2]> } : memref<?xf32>
    memref.store %val, %arg2[%arg3] { alias = #memref.aliasing<alias_scopes=[#alias_scope2], noalias=[#alias_scope1]> } : memref<?xf32>
    memref.alias_domain_scope.return
  }

  // CHECK-NOT: alias_domain_scope
  %0 = memref.alias_domain_scope -> f32 {
  // CHECK:  %[[VAL:.*]] = memref.load %{{.*}}[%{{.*}}] {alias = #memref.aliasing<alias_scopes = [#[[SCOPE2_1]]], noalias = [#[[SCOPE2_2]]]>} : memref<?xf32>
  // CHECK:  memref.store %[[VAL]], %{{.*}}[%{{.*}}] {alias = #memref.aliasing<alias_scopes = [#[[SCOPE2_2]]], noalias = [#[[SCOPE2_1]]]>} : memref<?xf32>
    %val = memref.load %arg1[%arg3] { alias = #memref.aliasing<alias_scopes=[#alias_scope1], noalias=[#alias_scope2]> } : memref<?xf32>
    memref.store %val, %arg2[%arg3] { alias = #memref.aliasing<alias_scopes=[#alias_scope2], noalias=[#alias_scope1]> } : memref<?xf32>
    memref.alias_domain_scope.return %val: f32
  }
  // CHECK: return %[[VAL]]
  return %0 : f32
}

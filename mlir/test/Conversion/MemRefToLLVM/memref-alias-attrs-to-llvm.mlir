// RUN: mlir-opt --convert-memref-alias-attributes-to-llvm %s | FileCheck %s

#alias_scope_domain = #memref.alias_scope_domain<id = distinct[0]<>, description = "The domain">
#alias_scope1 = #memref.alias_scope<id = distinct[1]<>, description = "scope">
#alias_scope2 = #memref.alias_scope<id = distinct[2]<>>

// CHECK: #[[DOMAIN:.*]] = #llvm.alias_scope_domain<id = distinct[0]<>, description = "The domain">
// CHECK-DAG: #[[SCOPE1:.*]] = #llvm.alias_scope<id = distinct[1]<>, domain = #[[DOMAIN]], description = "scope">
// CHECK-DAG: #[[SCOPE2:.*]] = #llvm.alias_scope<id = distinct[2]<>, domain = #[[DOMAIN]]>

// CHECK: func @memref_alias_attributes
func.func @memref_alias_attributes(%arg1 : memref<?xf32>, %arg2 : memref<?xf32>, %arg3: index) -> f32 {
  %0 = memref.alias_domain_scope #alias_scope_domain -> f32 {
  // CHECK:  %[[VAL:.*]] = memref.load %{{.*}}[%{{.*}}] {alias = #memref.aliasing<alias_scopes = [#[[SCOPE1]]], noalias = [#[[SCOPE2]]]>} : memref<?xf32>
  // CHECK:  memref.store %[[VAL]], %{{.*}}[%{{.*}}] {alias = #memref.aliasing<alias_scopes = [#[[SCOPE2]]], noalias = [#[[SCOPE1]]]>} : memref<?xf32>
    %val = memref.load %arg1[%arg3] { alias = #memref.aliasing<alias_scopes=[#alias_scope1], noalias=[#alias_scope2]> } : memref<?xf32>
    memref.store %val, %arg2[%arg3] { alias = #memref.aliasing<alias_scopes=[#alias_scope2], noalias=[#alias_scope1]> } : memref<?xf32>
    memref.alias_domain_scope.return %val: f32
  }
  return %0 : f32
}

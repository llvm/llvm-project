// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @foo(%arg0: !llvm.ptr)

#alias_scope_domain = #ptr.alias_scope_domain<id = distinct[0]<>, description = "The domain">
#alias_scope1 = #ptr.alias_scope<id = distinct[1]<>, domain = #alias_scope_domain, description = "The first scope">
#alias_scope2 = #ptr.alias_scope<id = distinct[2]<>, domain = #alias_scope_domain>
#alias_scope3 = #ptr.alias_scope<id = distinct[3]<>, domain = #alias_scope_domain>

// CHECK-LABEL: @alias_scopes
llvm.func @alias_scopes(%arg1 : !llvm.ptr) {
  %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK:  call void @llvm.experimental.noalias.scope.decl(metadata ![[SCOPES1:[0-9]+]])
  llvm.intr.experimental.noalias.scope.decl #alias_scope1
  // CHECK:  store {{.*}}, !alias.scope ![[SCOPES1]], !noalias ![[SCOPES23:[0-9]+]]
  ptr.store %0, %arg1 {alias_scopes = [#alias_scope1], noalias_scopes = [#alias_scope2, #alias_scope3]} : i32, !llvm.ptr
  // CHECK:  load {{.*}}, !alias.scope ![[SCOPES2:[0-9]+]], !noalias ![[SCOPES13:[0-9]+]]
  %1 = ptr.load %arg1 {alias_scopes = [#alias_scope2], noalias_scopes = [#alias_scope1, #alias_scope3]} : !llvm.ptr -> i32
  // CHECK:  atomicrmw {{.*}}, !alias.scope ![[SCOPES3:[0-9]+]], !noalias ![[SCOPES12:[0-9]+]]
  %2 = ptr.atomicrmw add %arg1, %0 monotonic {alias_scopes = [#alias_scope3], noalias_scopes = [#alias_scope1, #alias_scope2]} : !llvm.ptr, i32
  // CHECK:  cmpxchg {{.*}}, !alias.scope ![[SCOPES3]]
  %3, %4 = ptr.cmpxchg %arg1, %1, %2 acq_rel monotonic {alias_scopes = [#alias_scope3]} : !llvm.ptr, i32
  %5 = llvm.mlir.constant(42 : i8) : i8
  // CHECK:  llvm.memcpy{{.*}}, !alias.scope ![[SCOPES3]]
  "llvm.intr.memcpy"(%arg1, %arg1, %0) <{isVolatile = false}> {alias_scopes = [#alias_scope3]} : (!llvm.ptr, !llvm.ptr, i32) -> ()
  // CHECK:  llvm.memset{{.*}}, !noalias ![[SCOPES3]]
  "llvm.intr.memset"(%arg1, %5, %0) <{isVolatile = false}> {noalias_scopes = [#alias_scope3]} : (!llvm.ptr, i8, i32) -> ()
  // CHECK: call void @foo({{.*}} !alias.scope ![[SCOPES3]]
  llvm.call @foo(%arg1) {alias_scopes = [#alias_scope3]} : (!llvm.ptr) -> ()
  // CHECK: call void @foo({{.*}} !noalias ![[SCOPES3]]
  llvm.call @foo(%arg1) {noalias_scopes = [#alias_scope3]} : (!llvm.ptr) -> ()
  llvm.return
}

// Check the intrinsic declarations.
// CHECK-DAG: declare void @llvm.experimental.noalias.scope.decl(metadata)
// CHECK-DAG: declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
// CHECK-DAG: declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg)

// Check the translated metadata.
// CHECK-DAG: ![[DOMAIN:[0-9]+]] = distinct !{![[DOMAIN]], !"The domain"}
// CHECK-DAG: ![[SCOPE1:[0-9]+]] = distinct !{![[SCOPE1]], ![[DOMAIN]], !"The first scope"}
// CHECK-DAG: ![[SCOPE2:[0-9]+]] = distinct !{![[SCOPE2]], ![[DOMAIN]]}
// CHECK-DAG: ![[SCOPE3:[0-9]+]] = distinct !{![[SCOPE3]], ![[DOMAIN]]}
// CHECK-DAG: ![[SCOPES1]] = !{![[SCOPE1]]}
// CHECK-DAG: ![[SCOPES2]] = !{![[SCOPE2]]}
// CHECK-DAG: ![[SCOPES3]] = !{![[SCOPE3]]}
// CHECK-DAG: ![[SCOPES12]] = !{![[SCOPE1]], ![[SCOPE2]]}
// CHECK-DAG: ![[SCOPES13]] = !{![[SCOPE1]], ![[SCOPE3]]}
// CHECK-DAG: ![[SCOPES23]] = !{![[SCOPE2]], ![[SCOPE3]]}

// -----

// This test verifies the noalias scope intrinsice can be translated in
// isolation. It is the only operation using alias scopes attributes without
// implementing AliasAnalysisOpInterface.

#alias_scope_domain = #ptr.alias_scope_domain<id = distinct[0]<>, description = "The domain">
#alias_scope1 = #ptr.alias_scope<id = distinct[1]<>, domain = #alias_scope_domain>

// CHECK-LABEL: @noalias_intr_only
llvm.func @noalias_intr_only() {
  // CHECK: call void @llvm.experimental.noalias.scope.decl(metadata ![[SCOPES:[0-9]+]])
  llvm.intr.experimental.noalias.scope.decl #alias_scope1
  llvm.return
}

// Check the translated metadata.
// CHECK-DAG: ![[DOMAIN:[0-9]+]] = distinct !{![[DOMAIN]], !"The domain"}
// CHECK-DAG: ![[SCOPE:[0-9]+]] = distinct !{![[SCOPE]], ![[DOMAIN]]}
// CHECK-DAG: ![[SCOPES]] = !{![[SCOPE]]}

// -----

// This test ensures the alias scope translation creates a temporary metadata
// node as a placeholder for self-references. Without this, the debug info
// translation of a type list with a null entry could inadvertently reference
// access group metadata. This occurs when both translations generate a metadata
// list with a null entry, which are then uniqued to the same metadata node.
// The access group translation subsequently updates the null entry to a
// self-reference, which causes the type list to reference the access
// group node as well. The use of a temporary placeholder node avoids the issue.

#alias_scope_domain = #ptr.alias_scope_domain<id = distinct[0]<>>
#alias_scope = #ptr.alias_scope<id = distinct[1]<>, domain = #alias_scope_domain>

#di_null_type = #llvm.di_null_type
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_null_type>
#di_file = #llvm.di_file<"attribute-alias-scope.mlir" in "">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C11, file = #di_file, isOptimized = true, emissionKind = Full>
#di_subprogram = #llvm.di_subprogram<id = distinct[2]<>, compileUnit = #di_compile_unit, scope = #di_file, file = #di_file, subprogramFlags = "Definition", type = #di_subroutine_type>

// CHECK-LABEL: @self_reference
llvm.func @self_reference() {
  // CHECK: call void @llvm.experimental.noalias.scope.decl(metadata ![[SCOPES:[0-9]+]])
  llvm.intr.experimental.noalias.scope.decl #alias_scope
  llvm.return
} loc(fused<#di_subprogram>[unknown])

// Check that the translated subroutine types do not reference the access group
// domain since both of them are created as metadata list with a null entry.
// CHECK-DAG: ![[DOMAIN:[0-9]+]] = distinct !{![[DOMAIN]]}
// CHECK-DAG: ![[SCOPE:[0-9]+]] = distinct !{![[SCOPE]], ![[DOMAIN]]}
// CHECK-DAG: ![[SCOPES]] = !{![[SCOPE]]}
// CHECK-DAG: = !DISubroutineType(types: ![[TYPES:[0-9]+]])
// CHECK-DAG: ![[TYPES]] = !{null}

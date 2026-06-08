// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @function_metadata()
// CHECK-SAME: !type ![[TYPE:[0-9]+]]
// CHECK-SAME: !annotation ![[ANNOTATION:[0-9]+]]
llvm.func @function_metadata() attributes {
  function_metadata = [
    #llvm.func_metadata<"annotation", #llvm.md_node<
      #llvm.md_string<"function annotation">
    >>,
    #llvm.func_metadata<"type", #llvm.md_node<
      #llvm.md_const<0 : i64>,
      #llvm.md_string<"typeid">
    >>
  ]
} {
  llvm.return
}

// CHECK-DAG: ![[ANNOTATION]] = !{!"function annotation"}
// CHECK-DAG: ![[TYPE]] = !{i64 0, !"typeid"}

// -----

// CHECK-LABEL: declare !annotation
// CHECK-SAME: ![[DECL_ANNOTATION:[0-9]+]] void @declaration_metadata()
llvm.func @declaration_metadata() attributes {
  function_metadata = [
    #llvm.func_metadata<"annotation", #llvm.md_node<
      #llvm.md_string<"declaration annotation">
    >>
  ]
}

// CHECK-DAG: ![[DECL_ANNOTATION]] = !{!"declaration annotation"}

// -----

// CHECK-LABEL: define void @uses_later()
// CHECK-SAME: !callee ![[LATER_NODE:[0-9]+]]
llvm.func @uses_later() attributes {
  function_metadata = [
    #llvm.func_metadata<"callee", #llvm.md_node<#llvm.md_string<"later">, #llvm.md_value<@later>>>
  ]
} {
  llvm.return
}

llvm.func @later() {
  llvm.return
}

// CHECK-DAG: ![[LATER_NODE]] = !{!"later", ptr @later}

// -----

llvm.mlir.global internal @metadata_global(0 : i32) : i32

// CHECK-LABEL: define void @uses_global()
// CHECK-SAME: !callee ![[GLOBAL_NODE:[0-9]+]]
llvm.func @uses_global() attributes {
  function_metadata = [
    #llvm.func_metadata<"callee", #llvm.md_node<#llvm.md_value<@metadata_global>>>
  ]
} {
  llvm.return
}

// CHECK-DAG: ![[GLOBAL_NODE]] = !{ptr @metadata_global}

// -----

llvm.func @metadata_alias_target() {
  llvm.return
}

llvm.mlir.alias external @metadata_alias : !llvm.func<void ()> {
  %0 = llvm.mlir.addressof @metadata_alias_target : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: define void @uses_alias()
// CHECK-SAME: !callee ![[ALIAS_NODE:[0-9]+]]
llvm.func @uses_alias() attributes {
  function_metadata = [
    #llvm.func_metadata<"callee", #llvm.md_node<#llvm.md_value<@metadata_alias>>>
  ]
} {
  llvm.return
}

// CHECK-DAG: ![[ALIAS_NODE]] = !{ptr @metadata_alias}

// -----

llvm.mlir.ifunc external @metadata_ifunc : !llvm.func<void ()>, !llvm.ptr @metadata_ifunc_resolver

llvm.func @metadata_ifunc_resolver() -> !llvm.ptr {
  %0 = llvm.mlir.addressof @metadata_ifunc_target : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.func @metadata_ifunc_target() {
  llvm.return
}

// CHECK-LABEL: define void @uses_ifunc()
// CHECK-SAME: !callee ![[IFUNC_NODE:[0-9]+]]
llvm.func @uses_ifunc() attributes {
  function_metadata = [
    #llvm.func_metadata<"callee", #llvm.md_node<#llvm.md_value<@metadata_ifunc>>>
  ]
} {
  llvm.return
}

// CHECK-DAG: ![[IFUNC_NODE]] = !{ptr @metadata_ifunc}

// -----

// CHECK-LABEL: define void @repeated_kind_metadata()
// CHECK-SAME: !type ![[TYPE0:[0-9]+]]
// CHECK-SAME: !type ![[TYPE1:[0-9]+]]
llvm.func @repeated_kind_metadata() attributes {
  function_metadata = [
    #llvm.func_metadata<"type", #llvm.md_node<#llvm.md_const<0 : i64>, #llvm.md_string<"typeid0">>>,
    #llvm.func_metadata<"type", #llvm.md_node<#llvm.md_const<0 : i64>, #llvm.md_string<"typeid1">>>
  ]
} {
  llvm.return
}

// CHECK-DAG: ![[TYPE0]] = !{i64 0, !"typeid0"}
// CHECK-DAG: ![[TYPE1]] = !{i64 0, !"typeid1"}

// -----

// expected-error @below{{failed to convert function_metadata entry 'callee': could not resolve metadata reference '@missing'}}
llvm.func @missing_function_metadata_ref() attributes {
  function_metadata = [
    #llvm.func_metadata<"callee", #llvm.md_node<#llvm.md_value<@missing>>>
  ]
} {
  llvm.return
}

// -----

// expected-error @below{{failed to convert function_metadata entry 'bad': expected integer attribute in metadata constant}}
llvm.func @malformed_function_metadata() attributes {
  function_metadata = [
    #llvm.func_metadata<"bad", #llvm.md_node<#llvm.md_const<"not an integer">>>
  ]
} {
  llvm.return
}

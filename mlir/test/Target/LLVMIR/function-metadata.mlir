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

// Function metadata is converted after functions and ifuncs are mapped, so
// references to symbols declared later in the module can be resolved.
// CHECK-LABEL: define void @uses_later_symbols()
// CHECK-SAME: !refs ![[LATER_NODE:[0-9]+]]
llvm.func @uses_later_symbols() attributes {
  function_metadata = [
    #llvm.func_metadata<"refs", #llvm.md_node<
      #llvm.md_global_value<@later_function>,
      #llvm.md_global_value<@later_ifunc>
    >>
  ]
} {
  llvm.return
}

llvm.func @later_function() {
  llvm.return
}

llvm.mlir.ifunc external @later_ifunc : !llvm.func<void ()>, !llvm.ptr @later_ifunc_resolver

llvm.func @later_ifunc_resolver() -> !llvm.ptr {
  %0 = llvm.mlir.addressof @later_function : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-DAG: ![[LATER_NODE]] = !{ptr @later_function, ptr @later_ifunc}

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
    #llvm.func_metadata<"callee", #llvm.md_node<#llvm.md_global_value<@missing>>>
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

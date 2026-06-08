// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Tests LLVM named metadata translation with deeply nested metadata trees.

// CHECK: !foo.version = !{![[VERSION:[0-9]+]]}
// CHECK: !foo.language_version = !{![[LANG:[0-9]+]]}
// CHECK: !foo.kernel = !{![[KERNEL:[0-9]+]]}
// CHECK: !foo.global_refs = !{![[GLOBAL_REFS:[0-9]+]]}

llvm.func @my_kernel() {
  llvm.return
}

llvm.mlir.global internal @metadata_global(0 : i32) : i32

llvm.func @metadata_alias_target() {
  llvm.return
}

llvm.mlir.alias external @metadata_alias : !llvm.func<void ()> {
  %0 = llvm.mlir.addressof @metadata_alias_target : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.ifunc external @metadata_ifunc : !llvm.func<void ()>, !llvm.ptr @metadata_ifunc_resolver

llvm.func @metadata_ifunc_resolver() -> !llvm.ptr {
  %0 = llvm.mlir.addressof @metadata_ifunc_target : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.func @metadata_ifunc_target() {
  llvm.return
}

llvm.named_metadata "foo.version" [
  #llvm.md_node<#llvm.md_const<1 : i32>,
                #llvm.md_const<0 : i32>,
                #llvm.md_const<0 : i32>>
]
// CHECK-DAG: ![[VERSION]] = !{i32 1, i32 0, i32 0}

llvm.named_metadata "foo.language_version" [
  #llvm.md_node<#llvm.md_string<"Bar">,
                #llvm.md_const<1 : i32>,
                #llvm.md_const<2 : i32>,
                #llvm.md_const<3 : i32>>
]
// CHECK-DAG: ![[LANG]] = !{!"Bar", i32 1, i32 2, i32 3}

#buf0 = #llvm.md_node<
  #llvm.md_const<0 : i32>, #llvm.md_string<"foo.buffer">,
  #llvm.md_string<"foo.idx">, #llvm.md_const<0 : i32>,
  #llvm.md_const<1 : i32>, #llvm.md_string<"foo.read">,
  #llvm.md_string<"foo.address_space">, #llvm.md_const<1 : i32>,
  #llvm.md_string<"foo.size">, #llvm.md_const<4 : i32>,
  #llvm.md_string<"foo.align_size">, #llvm.md_const<4 : i32>>
// CHECK-DAG: ![[A0:[0-9]+]] = !{i32 0, !"foo.buffer", !"foo.idx", i32 0, i32 1, !"foo.read", !"foo.address_space", i32 1, !"foo.size", i32 4, !"foo.align_size", i32 4}

llvm.named_metadata "foo.kernel" [
  #llvm.md_node<
    #llvm.md_func<@my_kernel>,
    #llvm.md_node<>,
    #llvm.md_node<#buf0>>
]
// CHECK-DAG: ![[KERNEL]] = !{ptr @my_kernel, ![[EMPTY:[0-9]+]], ![[ARGS:[0-9]+]]}
// CHECK-DAG: ![[EMPTY]] = !{}
// CHECK-DAG: ![[ARGS]] = !{![[A0]]}

llvm.named_metadata "foo.global_refs" [
  #llvm.md_node<
    #llvm.md_func<@metadata_global>,
    #llvm.md_func<@metadata_alias>,
    #llvm.md_func<@metadata_ifunc>>
]
// CHECK-DAG: ![[GLOBAL_REFS]] = !{ptr @metadata_global, ptr @metadata_alias, ptr @metadata_ifunc}

// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Tests LLVM named metadata translation with deeply nested metadata trees.

// CHECK: !foo.version = !{![[VERSION:[0-9]+]]}
// CHECK: !foo.language_version = !{![[LANG:[0-9]+]]}
// CHECK: !foo.kernel = !{![[KERNEL:[0-9]+]]}

llvm.func @my_kernel() {
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

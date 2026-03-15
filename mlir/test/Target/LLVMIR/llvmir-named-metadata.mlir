// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Tests LLVM named metadata translation with deeply nested metadata trees.

// CHECK: !foo.version = !{![[VERSION:[0-9]+]]}
// CHECK: !foo.language_version = !{![[LANG:[0-9]+]]}
// CHECK: !foo.kernel = !{![[KERNEL:[0-9]+]]}

llvm.func @my_kernel() {
  llvm.return
}

llvm.named_metadata "foo.version" [
  #llvm.md_node<[#llvm.md_const<1 : i32>, #llvm.md_const<0 : i32>,
                  #llvm.md_const<0 : i32>]>
]
// CHECK-DAG: ![[VERSION]] = !{i32 1, i32 0, i32 0}

llvm.named_metadata "foo.language_version" [
  #llvm.md_node<[#llvm.md_string<"Bar">, #llvm.md_const<1 : i32>,
                  #llvm.md_const<2 : i32>, #llvm.md_const<3 : i32>]>
]
// CHECK-DAG: ![[LANG]] = !{!"Bar", i32 1, i32 2, i32 3}

#buf0 = #llvm.md_node<[
  #llvm.md_const<0 : i32>, #llvm.md_string<"foo.buffer">,
  #llvm.md_string<"foo.idx">, #llvm.md_const<0 : i32>,
  #llvm.md_const<1 : i32>, #llvm.md_string<"foo.read">,
  #llvm.md_string<"foo.address_space">, #llvm.md_const<1 : i32>,
  #llvm.md_string<"foo.size">, #llvm.md_const<4 : i32>,
  #llvm.md_string<"foo.align_size">, #llvm.md_const<4 : i32>
]>
// CHECK-DAG: ![[A0:[0-9]+]] = !{i32 0, !"foo.buffer", !"foo.idx", i32 0, i32 1, !"foo.read", !"foo.address_space", i32 1, !"foo.size", i32 4, !"foo.align_size", i32 4}

#buf1 = #llvm.md_node<[
  #llvm.md_const<1 : i32>, #llvm.md_string<"foo.buffer">,
  #llvm.md_string<"foo.idx">, #llvm.md_const<1 : i32>,
  #llvm.md_const<1 : i32>, #llvm.md_string<"foo.read_write">,
  #llvm.md_string<"foo.address_space">, #llvm.md_const<1 : i32>,
  #llvm.md_string<"foo.size">, #llvm.md_const<4 : i32>,
  #llvm.md_string<"foo.align_size">, #llvm.md_const<4 : i32>
]>
// CHECK-DAG: ![[A1:[0-9]+]] = !{i32 1, !"foo.buffer", !"foo.idx", i32 1, i32 1, !"foo.read_write", !"foo.address_space", i32 1, !"foo.size", i32 4, !"foo.align_size", i32 4}

#buf2 = #llvm.md_node<[
  #llvm.md_const<2 : i32>, #llvm.md_string<"foo.buffer">,
  #llvm.md_string<"foo.idx">, #llvm.md_const<2 : i32>,
  #llvm.md_const<1 : i32>, #llvm.md_string<"foo.read">,
  #llvm.md_string<"foo.address_space">, #llvm.md_const<2 : i32>,
  #llvm.md_string<"foo.size">, #llvm.md_const<4 : i32>,
  #llvm.md_string<"foo.align_size">, #llvm.md_const<4 : i32>
]>
// CHECK-DAG: ![[A2:[0-9]+]] = !{i32 2, !"foo.buffer", !"foo.idx", i32 2, i32 1, !"foo.read", !"foo.address_space", i32 2, !"foo.size", i32 4, !"foo.align_size", i32 4}

#pos3 = #llvm.md_node<[
  #llvm.md_const<3 : i32>, #llvm.md_string<"foo.block_position_in_grid">,
  #llvm.md_string<"foo.name">, #llvm.md_string<"vec3">,
  #llvm.md_string<"foo.arg_name">, #llvm.md_string<"block_position_in_grid">
]>
// CHECK-DAG: ![[A3:[0-9]+]] = !{i32 3, !"foo.block_position_in_grid", !"foo.name", !"vec3", !"foo.arg_name", !"block_position_in_grid"}

#pos4 = #llvm.md_node<[
  #llvm.md_const<4 : i32>, #llvm.md_string<"foo.blocks_per_grid">,
  #llvm.md_string<"foo.name">, #llvm.md_string<"vec3">,
  #llvm.md_string<"foo.arg_name">, #llvm.md_string<"blocks_per_grid">
]>
// CHECK-DAG: ![[A4:[0-9]+]] = !{i32 4, !"foo.blocks_per_grid", !"foo.name", !"vec3", !"foo.arg_name", !"blocks_per_grid"}

#pos5 = #llvm.md_node<[
  #llvm.md_const<5 : i32>, #llvm.md_string<"foo.thread_position_in_block">,
  #llvm.md_string<"foo.name">, #llvm.md_string<"vec3">,
  #llvm.md_string<"foo.arg_name">, #llvm.md_string<"thread_position_in_block">
]>
// CHECK-DAG: ![[A5:[0-9]+]] = !{i32 5, !"foo.thread_position_in_block", !"foo.name", !"vec3", !"foo.arg_name", !"thread_position_in_block"}

#pos6 = #llvm.md_node<[
  #llvm.md_const<6 : i32>, #llvm.md_string<"foo.threads_per_block">,
  #llvm.md_string<"foo.name">, #llvm.md_string<"vec3">,
  #llvm.md_string<"foo.arg_name">, #llvm.md_string<"threads_per_block">
]>
// CHECK-DAG: ![[A6:[0-9]+]] = !{i32 6, !"foo.threads_per_block", !"foo.name", !"vec3", !"foo.arg_name", !"threads_per_block"}

#pos7 = #llvm.md_node<[
  #llvm.md_const<7 : i32>, #llvm.md_string<"foo.thread_idx_in_warp">,
  #llvm.md_string<"foo.name">, #llvm.md_string<"vec1">,
  #llvm.md_string<"foo.arg_name">, #llvm.md_string<"thread_idx_in_warp">
]>
// CHECK-DAG: ![[A7:[0-9]+]] = !{i32 7, !"foo.thread_idx_in_warp", !"foo.name", !"vec1", !"foo.arg_name", !"thread_idx_in_warp"}

#pos8 = #llvm.md_node<[
  #llvm.md_const<8 : i32>, #llvm.md_string<"foo.warp_idx_in_block">,
  #llvm.md_string<"foo.name">, #llvm.md_string<"vec1">,
  #llvm.md_string<"foo.arg_name">, #llvm.md_string<"warp_idx_in_block">
]>
// CHECK-DAG: ![[A8:[0-9]+]] = !{i32 8, !"foo.warp_idx_in_block", !"foo.name", !"vec1", !"foo.arg_name", !"warp_idx_in_block"}

llvm.named_metadata "foo.kernel" [
  #llvm.md_node<[
    #llvm.md_func<@my_kernel>,
    #llvm.md_node<[]>,
    #llvm.md_node<[#buf0, #buf1, #buf2,
                    #pos3, #pos4, #pos5, #pos6, #pos7, #pos8]>
  ]>
]
// CHECK-DAG: ![[KERNEL]] = !{ptr @my_kernel, ![[EMPTY:[0-9]+]], ![[ARGS:[0-9]+]]}
// CHECK-DAG: ![[EMPTY]] = !{}
// CHECK-DAG: ![[ARGS]] = !{![[A0]], ![[A1]], ![[A2]], ![[A3]], ![[A4]], ![[A5]], ![[A6]], ![[A7]], ![[A8]]}

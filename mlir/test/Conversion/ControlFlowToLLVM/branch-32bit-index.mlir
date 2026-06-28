// RUN: mlir-opt %s --convert-cf-to-llvm -split-input-file | FileCheck %s

// Verify that ConvertControlFlowToLLVMPass respects the module's data layout
// when deriving the index type for block arguments.  When the module declares
// a 32-bit index via dlti.dl_spec, cf.br and cf.cond_br must pass index-typed
// block arguments as i32 instead of the default i64.

// -----

// 32-bit data layout: cf.br with index block argument -> i32 branch arg.

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {

func.func @cf_br_index_32bit(%arg0: index) -> index {
  cf.br ^bb1(%arg0 : index)
^bb1(%a: index):
  return %a : index
}

}

// CHECK-LABEL: func @cf_br_index_32bit
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i32
// CHECK: llvm.br ^{{.*}}(%{{.*}} : i32)
// CHECK: ^{{.*}}(%{{.*}}: i32):

// -----

// 32-bit data layout: cf.cond_br with index block arguments -> i32 branch args.

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {

func.func @cf_cond_br_index_32bit(%cond: i1, %a: index, %b: index) -> index {
  cf.cond_br %cond, ^bb1(%a : index), ^bb2(%b : index)
^bb1(%x: index):
  return %x : index
^bb2(%y: index):
  return %y : index
}

}

// CHECK-LABEL: func @cf_cond_br_index_32bit
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i32
// CHECK: llvm.cond_br %{{.*}}, ^{{.*}}(%{{.*}} : i32), ^{{.*}}(%{{.*}} : i32)

// -----

// Without dlti.dl_spec the default index width (i64) is preserved.

module {

func.func @cf_br_index_default(%arg0: index) -> index {
  cf.br ^bb1(%arg0 : index)
^bb1(%a: index):
  return %a : index
}

}

// CHECK-LABEL: func @cf_br_index_default
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i64
// CHECK: llvm.br ^{{.*}}(%{{.*}} : i64)
// CHECK: ^{{.*}}(%{{.*}}: i64):

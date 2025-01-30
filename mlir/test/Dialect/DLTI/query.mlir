// RUN: mlir-opt -transform-interpreter -canonicalize -split-input-file -verify-diagnostics %s | FileCheck %s

// expected-remark @below {{attr associated to "test.id" = 42 : i32}}
module attributes { test.dlti = #dlti.map<"test.id" = 42 : i32> } {
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %funcs : (!transform.any_op) -> !transform.any_op
    %param = transform.dlti.query ["test.id"] at %module : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %param, "attr associated to \"test.id\" =" at %module : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

// expected-remark @below {{i32 present in set : unit}}
module attributes { test.dlti = #dlti.map<i32 = unit> } {
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %funcs : (!transform.any_op) -> !transform.any_op
    %param = transform.dlti.query [i32] at %module : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %param, "i32 present in set :" at %module : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

// expected-remark @below {{attr associated to i32's "width_in_bits" = 32 : i32}}
module attributes { test.dlti = #dlti.map<i32 = #dlti.map<"width_in_bits" = 32 : i32>> } {
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %funcs : (!transform.any_op) -> !transform.any_op
    %param = transform.dlti.query [i32,"width_in_bits"] at %module : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %param, "attr associated to i32's \"width_in_bits\" =" at %module : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

// expected-remark @below {{width in bits of i32 = 32 : i64}}
// expected-remark @below {{width in bits of f64 = 64 : i64}}
module attributes { test.dlti = #dlti.map<"width_in_bits" = #dlti.map<i32 = 32, f64 = 64>> } {
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %funcs : (!transform.any_op) -> !transform.any_op
    %i32bits = transform.dlti.query ["width_in_bits",i32] at %module : (!transform.any_op) -> !transform.any_param
    %f64bits  = transform.dlti.query ["width_in_bits",f64] at %module : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %i32bits, "width in bits of i32 =" at %module : !transform.any_param, !transform.any_op
    transform.debug.emit_param_as_remark %f64bits, "width in bits of f64 =" at %module : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

// expected-remark @below {{attr associated to "test.id" = 42 : i32}}
module attributes { test.dlti = #dlti.dl_spec<"test.id" = 42 : i32> } {
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %funcs : (!transform.any_op) -> !transform.any_op
    %param = transform.dlti.query ["test.id"] at %module : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %param, "attr associated to \"test.id\" =" at %module : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

module attributes { test.dlti = #dlti.dl_spec<"test.id" = 42 : i32> } {
  // expected-remark @below {{attr associated to "test.id" = 24 : i32}}
  func.func private @f() attributes { test.dlti = #dlti.dl_spec<"test.id" = 24 : i32>}
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %param = transform.dlti.query ["test.id"] at %funcs : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %param, "attr associated to \"test.id\" =" at %funcs : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

// expected-remark @below {{attr associated to "test.id" = 42 : i32}}
module attributes { test.dlti = #dlti.dl_spec<"test.id" = 42 : i32> } {
  func.func private @f() attributes { test.dlti = #dlti.dl_spec<"test.id" = 24 : i32> }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %funcs : (!transform.any_op) -> !transform.any_op
    %param = transform.dlti.query ["test.id"] at %module : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %param, "attr associated to \"test.id\" =" at %module : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

module attributes { test.dlti = #dlti.dl_spec<"test.id" = 42 : i32> } {
  func.func @matmul_tensors(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
      -> tensor<?x?xf32> {
    // expected-remark @below {{associated attr 42 : i32}}
    %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                       outs(%arg2: tensor<?x?xf32>)
      -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg : (!transform.any_op) -> !transform.any_op
    %param = transform.dlti.query ["test.id"] at %matmul : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %param, "associated attr" at %matmul : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

module attributes { test.dlti = #dlti.dl_spec<"test.id" = 42 : i32> } {
  func.func @matmul_tensors(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
      -> tensor<?x?xf32> attributes {test.dlti = #dlti.dl_spec<"test.id" = 24 : i32> } {
    // expected-remark @below {{associated attr 24 : i32}}
    %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                       outs(%arg2: tensor<?x?xf32>)
      -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg : (!transform.any_op) -> !transform.any_op
    %param = transform.dlti.query ["test.id"] at %matmul : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %param, "associated attr" at %matmul : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

// expected-remark @below {{associated attr 42 : i32}}
module attributes { test.dlti =
  #dlti.target_system_spec<"CPU" =
    #dlti.target_device_spec<"test.id" = 42 : i32>> } {
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %func : (!transform.any_op) -> !transform.any_op
    %param = transform.dlti.query ["CPU","test.id"] at %module : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %param, "associated attr" at %module : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

module attributes { test.dlti = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"test.id" = 42 : i32>,
                                                         "GPU" = #dlti.target_device_spec<"test.id" = 43 : i32>> } {
  // expected-remark @below {{associated attr 43 : i32}}
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %param = transform.dlti.query ["GPU","test.id"] at %func : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %param, "associated attr" at %func : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

// Demonstation of nested lookup by walking ancestors and co-commitant shadowing.

// expected-remark @below {{associated CPU attr at module 42 : i32}}
// expected-remark @below {{associated GPU attr at module 43 : i32}}
module attributes { test.dlti = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"test.id" = 42 : i32>,
                                                         "GPU" = #dlti.target_device_spec<"test.id" = 43 : i32>> } {
  // expected-remark @below {{associated CPU attr at func 24 : i32}}
  // expected-remark @below {{associated GPU attr at func 43 : i32}}
  func.func private @f() attributes { test.dlti = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"test.id" = 24 : i32>> }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %func : (!transform.any_op) -> !transform.any_op
    // First the CPU attributes
    %cpu_func_param = transform.dlti.query ["CPU","test.id"] at %func : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %cpu_func_param, "associated CPU attr at func" at %func : !transform.any_param, !transform.any_op
    %cpu_module_param = transform.dlti.query ["CPU","test.id"] at %module : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %cpu_module_param, "associated CPU attr at module" at %module : !transform.any_param, !transform.any_op
    // Now the GPU attribute
    %gpu_func_param = transform.dlti.query ["GPU","test.id"] at %func : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %gpu_func_param, "associated GPU attr at func" at %func : !transform.any_param, !transform.any_op
    %gpu_module_param = transform.dlti.query ["GPU","test.id"] at %func : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %gpu_module_param , "associated GPU attr at module" at %module : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

module attributes { test.dlti = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"test.id" = 42 : i32>,
                                                         "GPU" = #dlti.target_device_spec<"test.id" = 43 : i32>> } {
  // expected-remark @below {{associated attr 43 : i32}}
  func.func private @f() attributes { test.dlti = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"test.id" = 24 : i32>> }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %param = transform.dlti.query ["GPU","test.id"] at %func : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %param, "associated attr" at %func : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

// Demonstation of nested lookup by walking ancestors and co-commitant shadowing.

// expected-remark @below {{associated CPU attr at module 42 : i32}}
// expected-remark @below {{associated GPU attr at module 43 : i32}}
module attributes { test.dlti = #dlti.map<"CPU" = #dlti.map<"test.id" = 42 : i32>,
                                          "GPU" = #dlti.map<"test.id" = 43 : i32>> } {
  // expected-remark @below {{associated CPU attr at func 42 : i32}}
  // expected-remark @below {{associated GPU attr at func 43 : i32}}
  func.func @f(%A: tensor<128x128xf32>) {
    // expected-remark @below {{associated CPU attr at matmul 24 : i32}}
    // expected-remark @below {{associated GPU attr at matmul 43 : i32}}
    %0 = linalg.matmul { test.dlti = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"test.id" = 24 : i32>> } ins(%A, %A : tensor<128x128xf32>, tensor<128x128xf32>)
                        outs(%A : tensor<128x128xf32>) -> tensor<128x128xf32>
    // expected-remark @below {{associated CPU attr at constant 42 : i32}}
    // expected-remark @below {{associated GPU attr at constant 43 : i32}}
    arith.constant 0 : i32
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %constant = transform.structured.match ops{["arith.constant"]} in %arg : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg : (!transform.any_op) -> !transform.any_op
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %func : (!transform.any_op) -> !transform.any_op
    // First query at the matmul
    %cpu_matmul_param = transform.dlti.query ["CPU","test.id"] at %matmul : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %cpu_matmul_param, "associated CPU attr at matmul" at %matmul : !transform.any_param, !transform.any_op
    %gpu_matmul_param = transform.dlti.query ["GPU","test.id"] at %matmul : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %gpu_matmul_param, "associated GPU attr at matmul" at %matmul : !transform.any_param, !transform.any_op
    // Now query at the constant
    %cpu_constant_param = transform.dlti.query ["CPU","test.id"] at %constant : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %cpu_constant_param, "associated CPU attr at constant" at %constant : !transform.any_param, !transform.any_op
    %gpu_constant_param = transform.dlti.query ["GPU","test.id"] at %constant : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %gpu_constant_param, "associated GPU attr at constant" at %constant : !transform.any_param, !transform.any_op
    // Now query at the func
    %cpu_func_param = transform.dlti.query ["CPU","test.id"] at %func : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %cpu_func_param, "associated CPU attr at func" at %func : !transform.any_param, !transform.any_op
    %gpu_func_param = transform.dlti.query ["GPU","test.id"] at %func : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %gpu_func_param, "associated GPU attr at func" at %func : !transform.any_param, !transform.any_op
    // Now query at the module
    %cpu_module_param = transform.dlti.query ["CPU","test.id"] at %module : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %cpu_module_param, "associated CPU attr at module" at %module : !transform.any_param, !transform.any_op
    %gpu_module_param = transform.dlti.query ["GPU","test.id"] at %module : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %gpu_module_param, "associated GPU attr at module" at %module : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// -----

module attributes { test.dlti = #dlti.target_system_spec<
  "CPU" = #dlti.target_device_spec<
    "cache::L1::size_in_bytes" = 65536 : i32,
    "cache::L1d::size_in_bytes" = 32768 : i32>> } {
  // expected-remark @below {{L1::size_in_bytes 65536 : i32}}
  // expected-remark @below {{L1d::size_in_bytes 32768 : i32}}
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %l1_size = transform.dlti.query ["CPU","cache::L1::size_in_bytes"] at %func : (!transform.any_op) -> !transform.param<i32>
    %l1d_size = transform.dlti.query ["CPU","cache::L1d::size_in_bytes"] at %func : (!transform.any_op) -> !transform.param<i32>
    transform.debug.emit_param_as_remark %l1_size, "L1::size_in_bytes" at %func : !transform.param<i32>, !transform.any_op
    transform.debug.emit_param_as_remark %l1d_size, "L1d::size_in_bytes" at %func : !transform.param<i32>, !transform.any_op
    transform.yield
  }
}

// -----

#l1_size = #dlti.map<"size_in_bytes" = 65536 : i32>
#l1d_size = #dlti.map<"size_in_bytes" = 32768 : i32>
module attributes { test.dlti =
  #dlti.target_system_spec<"CPU" =
    #dlti.target_device_spec<"cache" =
      #dlti.map<"L1" = #l1_size,
                "L1d" = #l1d_size >>> } {
  // expected-remark @below {{L1::size_in_bytes 65536 : i32}}
  // expected-remark @below {{L1d::size_in_bytes 32768 : i32}}
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    %l1_size = transform.dlti.query ["CPU","cache","L1","size_in_bytes"] at %func : (!transform.any_op) -> !transform.param<i32>
    %l1d_size = transform.dlti.query ["CPU","cache","L1d","size_in_bytes"] at %func : (!transform.any_op) -> !transform.param<i32>
    transform.debug.emit_param_as_remark %l1_size, "L1::size_in_bytes" at %func : !transform.param<i32>, !transform.any_op
    transform.debug.emit_param_as_remark %l1d_size, "L1d::size_in_bytes" at %func : !transform.param<i32>, !transform.any_op
    transform.yield
  }
}

// -----

module attributes { test.dlti = #dlti.target_system_spec<
  "CPU" = #dlti.target_device_spec<"inner_most_tile_size" = 42 : i32>> } {
  // CHECK-LABEL: func @matmul_tensors
  func.func @matmul_tensors(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
      -> tensor<?x?xf32> {
    // CHECK: scf.for {{.*}} to {{.*}} step {{.*}}42
    // CHECK:   tensor.extract_slice
    // CHECK:   linalg.matmul
    // CHECK:   tensor.insert_slice
    // CHECK:   scf.yield
    %0 = linalg.matmul ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                       outs(%arg2: tensor<?x?xf32>)
      -> tensor<?x?xf32>
    // CHECK: return
    return %0 : tensor<?x?xf32>
  }
}

// Demonstrates obtaining transform op parameters from DLTI attributes and directly putting them to use.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg : (!transform.any_op) -> !transform.any_op
    %tile_size = transform.dlti.query ["CPU","inner_most_tile_size"] at %matmul : (!transform.any_op) -> !transform.param<i32>
    transform.structured.tile_using_for %matmul tile_sizes [%tile_size] : (!transform.any_op, !transform.param<i32>) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// expected-note @below {{key not present - failed at keys: ["NPU"]}}
module attributes { test.dlti = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<"test.id" = 42 : i32>,
    "GPU" = #dlti.target_device_spec<"test.id" = 43 : i32>> } {
  // expected-error @below {{target op of failed DLTI query}}
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{'transform.dlti.query' op failed to apply}}
    %param = transform.dlti.query ["NPU","test.id"] at %func : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// -----

// expected-note @below {{key not present - failed at keys: ["CPU","unspecified"]}}
module attributes { test.dlti = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<"test.id" = 42 : i32>,
    "GPU" = #dlti.target_device_spec<"test.id" = 43 : i32>> } {
  // expected-error @below {{target op of failed DLTI query}}
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{'transform.dlti.query' op failed to apply}}
    %param = transform.dlti.query ["CPU","unspecified"] at %func : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// -----

// expected-note @below {{key not present - failed at keys: ["test.id"]}}
module attributes { test.dlti = #dlti.target_system_spec<
  "CPU" = #dlti.target_device_spec<"test.id" = 42 : i32>,
  "GPU" = #dlti.target_device_spec<"test.id" = 43 : i32>> } {
  // expected-error @below {{target op of failed DLTI query}}
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{'transform.dlti.query' op failed to apply}}
    %param = transform.dlti.query ["test.id"] at %func : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// -----

// expected-note @below {{key not present - failed at keys: ["CPU"]}}
module attributes { test.dlti = #dlti.dl_spec<"test.id" = 42 : i32> } {
  // expected-error @below {{target op of failed DLTI query}}
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{'transform.dlti.query' op failed to apply}}
    %param = transform.dlti.query ["CPU","test.id"] at %func : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// -----

// expected-note @below {{attribute at keys ["CPU"] is not queryable}}
module attributes { test.dlti = #dlti.dl_spec<"CPU" = 42 : i32> } {
  // expected-error @below {{target op of failed DLTI query}}
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{'transform.dlti.query' op failed to apply}}
    %param = transform.dlti.query ["CPU","test.id"] at %func : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// -----

// expected-note @below {{attribute at keys [i32] is not queryable}}
module attributes { test.dlti = #dlti.dl_spec<i32 = 32 : i32> } {
  // expected-error @below {{target op of failed DLTI query}}
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{'transform.dlti.query' op failed to apply}}
    %param = transform.dlti.query [i32,"width_in_bits"] at %func : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// -----

module {
  // expected-error @below {{target op of failed DLTI query}}
  // expected-note @below {{no DLTI-queryable attrs on target op or any of its ancestors}}
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{'transform.dlti.query' op failed to apply}}
    %param = transform.dlti.query ["CPU","test.id"] at %func : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// -----

// expected-note @below {{key not present - failed at keys: ["width_in_bits",i64]}}
module attributes { test.dlti = #dlti.map<"width_in_bits" = #dlti.map<i32 = 32>>} {
  // expected-error @below {{target op of failed DLTI query}}
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{'transform.dlti.query' op failed to apply}}
    %param = transform.dlti.query ["width_in_bits",i64] at %func : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// -----

module attributes { test.dlti = #dlti.dl_spec<"test.id" = 42 : i32>} {
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{'transform.dlti.query' keys of wrong type: only StringAttr and TypeAttr are allowed}}
    %param = transform.dlti.query [1] at %funcs : (!transform.any_op) -> !transform.param<i64>
    transform.yield
  }
}

// -----

module attributes { test.dlti = #dlti.map<#dlti.dl_entry<"test.id", 42 : i32>>} {
  // expected-error @below {{target op of failed DLTI query}}
  // expected-note @below {{no keys provided to attempt query with}}
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{'transform.dlti.query' op failed to apply}}
    %param = transform.dlti.query [] at %func : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// -----

module attributes { test.dlti = #dlti.dl_spec<#dlti.dl_entry<"test.id", 42 : i32>>} {
  func.func private @f()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected the type of the parameter attribute ('i32') to match the parameter type ('i64')}}
    %param = transform.dlti.query ["test.id"] at %funcs : (!transform.any_op) -> !transform.param<i64>
    transform.yield
  }
}

// -----

// expected-note @below {{attribute at keys ["CPU","test"] is not queryable}}
module attributes { test.dlti = #dlti.map<"CPU" = #dlti.map<"test" = {"id" = 0}>> } {
  // expected-note @below {{key not present - failed at keys: ["CPU","test","id"]}}
  func.func @f(%A: tensor<128x128xf32>) attributes { test.dlti = #dlti.map<"CPU" = #dlti.map<"test" = #dlti.map<"ego" = 0>>> } {
    scf.execute_region { // NB: No notes/errors on this unannotated ancestor
      // expected-note @below {{key not present - failed at keys: ["CPU","test"]}}
      // expected-error @below {{target op of failed DLTI query}}
      %0 = linalg.matmul { test.dlti = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"test.id" = 24 : i32>> } ins(%A, %A : tensor<128x128xf32>, tensor<128x128xf32>)
                          outs(%A : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield
    }
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{'transform.dlti.query' op failed to apply}}
    %cpu_matmul_param = transform.dlti.query ["CPU","test","id"] at %matmul : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(gpu.func(acc-cg-to-gpu{device-type=nvidia})))" \
// RUN:   --remarks-filter="(open)?acc.*" 2>&1 | FileCheck %s

// A routine private placed in a workgroup slot is reported as shared memory.

module attributes {gpu.container_module} {
  acc.routine @routine_vector func(@vector_scalar) vector
  acc.routine @routine_unnamed func(@unnamed) vector

  gpu.module @cuda_device_mod {
    func.func private @use(memref<f64>)

    // CHECK-NOT: GPU global memory used for
    // CHECK: Function=vector_scalar | Remark="GPU shared memory used for s"
    // CHECK-NOT: GPU global memory used for
    // CHECK-NOT: GPU shared memory used for
    gpu.func @vector_scalar() attributes {acc.specialized_routine =
        #acc.specialized_routine<@routine_vector, <vector>, "vector_scalar">} {
      %c32 = arith.constant 32 : index
      %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
      %priv = acc.privatize {acc.active_par_dims = #acc<active_par_dims[]>,
                             acc.var_name = #acc.var_name<"s">}
          : () -> !acc.private_type<memref<f64>>
      acc.compute_region launch(%arg0 = %tx) ins(%arg1 = %priv)
          : (!acc.private_type<memref<f64>>) {
        %local = acc.private_local %arg1
            {acc.var_name = #acc.var_name<"s">,
             acc.active_par_dims = #acc<active_par_dims[]>}
            : (!acc.private_type<memref<f64>>) -> memref<f64>
        acc.predicate_region {
          func.call @use(%local) : (memref<f64>) -> ()
        }
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }

    gpu.func @unnamed() attributes {acc.specialized_routine =
        #acc.specialized_routine<@routine_unnamed, <vector>, "unnamed">} {
      %c32 = arith.constant 32 : index
      %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
      %priv = acc.privatize {acc.active_par_dims = #acc<active_par_dims[]>}
          : () -> !acc.private_type<memref<f64>>
      acc.compute_region launch(%arg0 = %tx) ins(%arg1 = %priv)
          : (!acc.private_type<memref<f64>>) {
        %local = acc.private_local %arg1 {acc.active_par_dims = #acc<active_par_dims[]>}
            : (!acc.private_type<memref<f64>>) -> memref<f64>
        acc.predicate_region {
          func.call @use(%local) : (memref<f64>) -> ()
        }
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}

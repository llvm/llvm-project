// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(gpu.func(acc-cg-to-gpu{max-thread-private-stack=16384})))" \
// RUN:   --remarks-filter="(open)?acc.*" 2>&1 | FileCheck %s

module attributes {gpu.container_module} {
  acc.routine @routine_one func(@one_array) vector
  acc.routine @routine_two func(@two_arrays) vector
  acc.routine @routine_dyn func(@dynamic_extent) vector
  acc.routine @routine_unnamed func(@unnamed) vector

  gpu.module @cuda_device_mod {
    // CHECK: Function=one_array | Remark="GPU global memory used for buf"
    gpu.func @one_array() attributes {acc.specialized_routine =
        #acc.specialized_routine<@routine_one, <vector>, "one_array">} {
      %c32 = arith.constant 32 : index
      %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
      %priv = acc.privatize {acc.active_par_dims = #acc<active_par_dims[thread_x]>,
                             acc.var_name = #acc.var_name<"buf">}
          : () -> !acc.private_type<memref<4096xi32>>

      acc.compute_region launch(%arg0 = %tx) ins(%priv_in = %priv)
          : (!acc.private_type<memref<4096xi32>>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : i32
        %local = acc.private_local %priv_in
            {acc.var_name = #acc.var_name<"buf">,
             acc.active_par_dims = #acc<active_par_dims[thread_x]>}
            : (!acc.private_type<memref<4096xi32>>) -> memref<4096xi32>
        memref.store %c1, %local[%c0] : memref<4096xi32>
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }

    // CHECK: Function=two_arrays | Remark="GPU global memory used for a,b"
    gpu.func @two_arrays() attributes {acc.specialized_routine =
        #acc.specialized_routine<@routine_two, <vector>, "two_arrays">} {
      %c32 = arith.constant 32 : index
      %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
      %priv_a = acc.privatize {acc.active_par_dims = #acc<active_par_dims[thread_x]>,
                               acc.var_name = #acc.var_name<"a">}
          : () -> !acc.private_type<memref<4096xi32>>
      %priv_b = acc.privatize {acc.active_par_dims = #acc<active_par_dims[thread_x]>,
                               acc.var_name = #acc.var_name<"b">}
          : () -> !acc.private_type<memref<8192xi32>>

      acc.compute_region launch(%arg0 = %tx) ins(%a_in = %priv_a, %b_in = %priv_b)
          : (!acc.private_type<memref<4096xi32>>, !acc.private_type<memref<8192xi32>>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : i32
        %a = acc.private_local %a_in
            {acc.var_name = #acc.var_name<"a">,
             acc.active_par_dims = #acc<active_par_dims[thread_x]>}
            : (!acc.private_type<memref<4096xi32>>) -> memref<4096xi32>
        %b = acc.private_local %b_in
            {acc.var_name = #acc.var_name<"b">,
             acc.active_par_dims = #acc<active_par_dims[thread_x]>}
            : (!acc.private_type<memref<8192xi32>>) -> memref<8192xi32>
        memref.store %c1, %a[%c0] : memref<4096xi32>
        memref.store %c1, %b[%c0] : memref<8192xi32>
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }

    // CHECK: Function=dynamic_extent | Remark="GPU global memory used for work"
    gpu.func @dynamic_extent(%n: index) attributes {acc.specialized_routine =
        #acc.specialized_routine<@routine_dyn, <vector>, "dynamic_extent">} {
      %c32 = arith.constant 32 : index
      %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
      %priv = acc.privatize(%n) {acc.active_par_dims = #acc<active_par_dims[thread_x]>,
                                 acc.var_name = #acc.var_name<"work">}
          : (index) -> !acc.private_type<memref<?xi32>>

      acc.compute_region launch(%arg0 = %tx) ins(%priv_in = %priv)
          : (!acc.private_type<memref<?xi32>>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : i32
        %local = acc.private_local %priv_in
            {acc.var_name = #acc.var_name<"work">,
             acc.active_par_dims = #acc<active_par_dims[thread_x]>}
            : (!acc.private_type<memref<?xi32>>) -> memref<?xi32>
        memref.store %c1, %local[%c0] : memref<?xi32>
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }

    // CHECK-NOT: GPU global memory used for
    gpu.func @unnamed() attributes {acc.specialized_routine =
        #acc.specialized_routine<@routine_unnamed, <vector>, "unnamed">} {
      %c32 = arith.constant 32 : index
      %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
      %priv = acc.privatize {acc.active_par_dims = #acc<active_par_dims[thread_x]>}
          : () -> !acc.private_type<memref<4096xi32>>

      acc.compute_region launch(%arg0 = %tx) ins(%priv_in = %priv)
          : (!acc.private_type<memref<4096xi32>>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : i32
        %local = acc.private_local %priv_in
            {acc.active_par_dims = #acc<active_par_dims[thread_x]>}
            : (!acc.private_type<memref<4096xi32>>) -> memref<4096xi32>
        memref.store %c1, %local[%c0] : memref<4096xi32>
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}

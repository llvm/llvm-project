// RUN: mlir-opt --omp-host-op-filter %s | FileCheck %s
module attributes {omp.is_target_device = true} {
  // CHECK-LABEL: llvm.func @external() attributes {sym_visibility = "private"}
  // CHECK-NOT: llvm.return
  llvm.func @external() attributes {sym_visibility = "private"}

  // CHECK-LABEL: llvm.func @basic_checks
  // CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr, %[[PLACEHOLDER:.*]]: !llvm.ptr)
  llvm.func @basic_checks(%arg0: !llvm.ptr) -> !llvm.struct<(i32, f32)> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(10 : i32) : i32
    %3 = llvm.mlir.constant(2.500000e+00 : f32) : f32

    llvm.call @foo() : () -> ()

    // CHECK-NEXT: %[[GLOBAL:.*]] = llvm.mlir.addressof @global_scalar : !llvm.ptr
    %4 = llvm.mlir.addressof @global_scalar : !llvm.ptr
    %5 = llvm.mlir.constant(1 : i64) : i64

    // CHECK-NEXT: %[[HDA:.*]] = omp.map.info var_ptr(%[[PLACEHOLDER]]{{.*}})
    // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[ARG]]{{.*}})
    // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[GLOBAL]]{{.*}})
    // CHECK-NEXT: %[[MAP2:.*]] = omp.map.info var_ptr(%[[PLACEHOLDER]]{{.*}})
    %6 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %7 = omp.map.info var_ptr(%4 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %8 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %9 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr

    // CHECK-NEXT: omp.target kernel_type(generic) has_device_addr(%[[HDA]] -> {{.*}} : {{.*}}) map_entries(%[[MAP0]] -> {{.*}}, %[[MAP1]] -> {{.*}}, %[[MAP2]] -> {{.*}} : {{.*}})
    omp.target kernel_type(generic) has_device_addr(%8 -> %arg1 : !llvm.ptr) map_entries(%6 -> %arg2, %7 -> %arg3, %9 -> %arg4 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      // CHECK-NEXT: llvm.call
      llvm.call @foo() : () -> ()
      omp.terminator
    }

    // CHECK-NOT: omp.parallel
    // CHECK-NOT: llvm.call
    omp.parallel {
      llvm.call @foo() : () -> ()
      omp.terminator
    }

    // CHECK-NOT: omp.map.info
    %10 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %11 = omp.map.info var_ptr(%4 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %12 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %13 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr

    // CHECK-NOT: omp.target_data
    omp.target_data map_entries(%10, %11, %12, %13 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      // CHECK-NOT: llvm.call
      llvm.call @foo() : () -> ()
      omp.terminator
    }

    // CHECK-NOT: omp.target_enter_data
    // CHECK-NOT: omp.target_exit_data
    // CHECK-NOT: omp.target_update
    %14 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr
    omp.target_enter_data map_entries(%14 : !llvm.ptr)
    %15 = omp.map.info var_ptr(%4 : !llvm.ptr, i32) map_clauses(from) capture(ByRef) -> !llvm.ptr
    omp.target_exit_data map_entries(%15 : !llvm.ptr)
    %16 = omp.map.info var_ptr(%1 : !llvm.ptr, !llvm.ptr) map_clauses(to) capture(ByRef) -> !llvm.ptr
    omp.target_update map_entries(%16 : !llvm.ptr)

    // CHECK-NOT: llvm.call
    llvm.call @foo() : () -> ()

    // CHECK: {{llvm\.return$}}
    %17 = llvm.mlir.poison : !llvm.struct<(i32, f32)>
    %18 = llvm.insertvalue %2, %17[0] : !llvm.struct<(i32, f32)>
    %19 = llvm.insertvalue %3, %18[1] : !llvm.struct<(i32, f32)>
    llvm.return %19 : !llvm.struct<(i32, f32)>
  }

  // CHECK-LABEL: llvm.func @member_map_complex_init
  // CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr, %[[PLACEHOLDER:.*]]: !llvm.ptr)
  llvm.func @member_map_complex_init(%arg0: !llvm.ptr) {
    // CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[PLACEHOLDER]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<{{.*}}>
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<15 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<15 x array<3 x i64>>)> : (i64) -> !llvm.ptr
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<15 x array<3 x i64>>)> : (i64) -> !llvm.ptr
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(0 : i8) : i8
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.zero : !llvm.ptr
    llvm.call @LangRTPlaceholderFunc(%5, %arg0, %10, %7, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i8, i32) -> ()
    %11 = llvm.mlir.constant(384 : i32) : i32
    "llvm.intr.memcpy"(%1, %5, %11) <{arg_attrs = [{llvm.align = 8 : i64}, {llvm.align = 8 : i64}, {}], isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    %12 = llvm.mlir.constant(24 : i32) : i32
    %13 = llvm.getelementptr %1[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<15 x array<3 x i64>>)>
    %14 = llvm.load %13 : !llvm.ptr -> i8
    %15 = llvm.sext %14 : i8 to i32
    %16 = llvm.mlir.constant(24 : i32) : i32
    %17 = llvm.mul %16, %15 : i32
    %18 = llvm.add %12, %17 : i32
    "llvm.intr.memcpy"(%3, %1, %18) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    %19 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<15 x array<3 x i64>>)>

    // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[PLACEHOLDER]] {{.*}} map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[GEP]] : {{.*}})
    // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[PLACEHOLDER]] {{.*}} map_clauses(to) capture(ByRef) members(%[[MAP0]] : [0] : !llvm.ptr)
    %20 = omp.map.info var_ptr(%3 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<15 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%19 : !llvm.ptr, f32) -> !llvm.ptr
    %21 = omp.map.info var_ptr(%3 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<15 x array<3 x i64>>)>) map_clauses(to) capture(ByRef) members(%20 : [0] : !llvm.ptr) -> !llvm.ptr

    // CHECK-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP1]] -> %{{.*}}, %[[MAP0]] -> %{{.*}} : {{.*}})
    omp.target kernel_type(generic) map_entries(%21 -> %arg1, %20 -> %arg2 : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }

  // CHECK-LABEL: llvm.func @target_data
  // CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: !llvm.ptr)
  llvm.func @target_data(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %1 = omp.map.info var_ptr(%arg1 : !llvm.ptr, i32) map_clauses(return_param) capture(ByRef) -> !llvm.ptr
    %2 = omp.map.info var_ptr(%arg2 : !llvm.ptr, !llvm.struct<"MyStruct", (i64)>) map_clauses(return_param) capture(ByRef) -> !llvm.ptr

    // CHECK-NOT: omp.target_data
    omp.target_data map_entries(%0 : !llvm.ptr) use_device_addr(%1 -> %arg3 : !llvm.ptr) use_device_ptr(%2 -> %arg4 : !llvm.ptr) {
      // CHECK:      %[[MAP0:.*]] = omp.map.info var_ptr(%[[ARG0]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef)
      // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[ARG1]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef)
      // CHECK-NEXT: %[[MAP2:.*]] = omp.map.info var_ptr(%[[ARG2]] : !llvm.ptr, !llvm.struct<"MyStruct", (i64)>) map_clauses(tofrom) capture(ByRef)
      %3 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
      %4 = omp.map.info var_ptr(%arg3 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
      %5 = omp.map.info var_ptr(%arg4 : !llvm.ptr, !llvm.struct<"MyStruct", (i64)>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr

      // CHECK-NOT: llvm.call
      llvm.call @foo() : () -> ()

      // CHECK: omp.target kernel_type(generic) map_entries(%[[MAP0]] -> %{{.*}}, %[[MAP1]] -> %{{.*}}, %[[MAP2]] -> %{{.*}} : !llvm.ptr, !llvm.ptr, !llvm.ptr)
      omp.target kernel_type(generic) map_entries(%3 -> %arg5, %4 -> %arg6, %5 -> %arg7 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
        omp.terminator
      }

      // CHECK-NOT: llvm.call
      llvm.call @foo() : () -> ()
      omp.terminator
    }

    // CHECK: llvm.return
    llvm.return
  }

  // CHECK-LABEL: llvm.func @no_target
  // CHECK-SAME: (%{{.*}}: !llvm.ptr)
  llvm.func @no_target(%arg0: !llvm.ptr) {
    // CHECK-NEXT: llvm.return
    %0 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target_data map_entries(%0 : !llvm.ptr) {
      %1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
      omp.target_data map_entries(%1 : !llvm.ptr) {
        llvm.call @foo() : () -> ()
        omp.terminator
      }
      llvm.call @foo() : () -> ()
      omp.terminator
    }
    llvm.return
  }

  // CHECK-LABEL: llvm.func @map_info_members
  // CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr)
  llvm.func @map_info_members(%arg0: !llvm.ptr) {
    // CHECK-NEXT: %[[VAR_PTR_PTR:.*]] = llvm.getelementptr %[[ARG]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(9 : index) : i64
    %5 = llvm.mlir.constant(48 : i32) : i32
    "llvm.intr.memcpy"(%1, %arg0, %5) <{arg_attrs = [{llvm.align = 8 : i64}, {llvm.align = 8 : i64}, {}], isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    %6 = llvm.getelementptr %1[0, 7, %2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %7 = llvm.load %6 : !llvm.ptr -> i64
    %8 = llvm.getelementptr %1[0, 7, %2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %9 = llvm.load %8 : !llvm.ptr -> i64
    %10 = llvm.getelementptr %1[0, 7, %2, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %11 = llvm.load %10 : !llvm.ptr -> i64
    %12 = llvm.sub %3, %7 : i64
    %13 = llvm.sub %4, %7 : i64
    %14 = omp.map.bounds lower_bound(%12 : i64) upper_bound(%13 : i64) extent(%9 : i64) stride(%11 : i64) start_idx(%7 : i64) {stride_in_bytes = true}
    %15 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>

    // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[ARG]] {{.*}} map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[VAR_PTR_PTR]] : !llvm.ptr, f32) -> !llvm.ptr
    // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[ARG]] {{.*}} map_clauses(to) capture(ByRef) members(%[[MAP0]] : [0] : !llvm.ptr) -> !llvm.ptr
    %16 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%15 : !llvm.ptr, f32) bounds(%14) -> !llvm.ptr
    %17 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(to) capture(ByRef) members(%16 : [0] : !llvm.ptr) -> !llvm.ptr

    // CHECK-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP1]] -> {{.*}}, %[[MAP0]] -> {{.*}} : !llvm.ptr, !llvm.ptr)
    omp.target kernel_type(generic) map_entries(%17 -> %arg1, %16 -> %arg2 : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }

    // CHECK: llvm.return
    llvm.return
  }

  // CHECK-LABEL: llvm.func @control_flow
  // CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr)
  llvm.func @control_flow(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[ARG0]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[ARG0]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %0 = llvm.load %arg1 : !llvm.ptr -> i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.icmp "ne" %0, %1 : i32
    llvm.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.call @foo() : () -> ()
    %3 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr

    // CHECK-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP0]] -> %{{.*}} : !llvm.ptr)
    omp.target kernel_type(generic) map_entries(%3 -> %arg2 : !llvm.ptr) {
      omp.terminator
    }
    llvm.call @foo() : () -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    // CHECK-NOT: llvm.call
    // CHECK-NOT: omp.map.info
    // CHECK-NOT: omp.target_data
    llvm.call @foo() : () -> ()

    %4 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target_data map_entries(%4 : !llvm.ptr) {
      llvm.call @foo() : () -> ()
      %5 = llvm.load %arg1 : !llvm.ptr -> i32
      %6 = llvm.mlir.constant(0 : i32) : i32
      %7 = llvm.icmp "ne" %5, %6 : i32
      llvm.cond_br %7, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      llvm.call @foo() : () -> ()
      %8 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr

      // CHECK: omp.target kernel_type(generic) map_entries(%[[MAP1]] -> %{{.*}} : !llvm.ptr)
      omp.target kernel_type(generic) map_entries(%8 -> %arg2 : !llvm.ptr) {
        omp.terminator
      }

      // CHECK-NOT: llvm.call
      // CHECK-NOT: llvm.br
      llvm.call @foo() : () -> ()
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.call @foo() : () -> ()
      omp.terminator
    }
    llvm.call @foo() : () -> ()

    // CHECK: llvm.return
    llvm.return
  }

  // CHECK-LABEL: llvm.func @block_args
  // CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr, %[[PLACEHOLDER0:.*]]: !llvm.ptr, %[[PLACEHOLDER1:.*]]: !llvm.ptr)
  llvm.func @block_args(%arg0: !llvm.ptr) {
    // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[PLACEHOLDER0]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[PLACEHOLDER1]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.parallel private(@privatizer %arg0 -> %arg1 : !llvm.ptr) {
      %0 = omp.map.info var_ptr(%arg1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr

      // CHECK-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP0]] -> %{{.*}} : !llvm.ptr)
      omp.target kernel_type(generic) map_entries(%0 -> %arg2 : !llvm.ptr) {
        omp.terminator
      }
      omp.terminator
    }

    // CHECK-NOT: omp.parallel
    // CHECK-NOT: omp.map.info
    // CHECK-NOT: omp.target_data
    omp.parallel private(@privatizer %arg0 -> %arg1 : !llvm.ptr) {
      %0 = omp.map.info var_ptr(%arg1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
      omp.target_data map_entries(%0 : !llvm.ptr) {
        omp.parallel private(@privatizer %arg1 -> %arg2 : !llvm.ptr) {
          %1 = omp.map.info var_ptr(%arg2 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr

          // CHECK: omp.target kernel_type(generic) map_entries(%[[MAP1]] -> %{{.*}} : !llvm.ptr)
          omp.target kernel_type(generic) map_entries(%1 -> %arg3 : !llvm.ptr) {
            omp.terminator
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    // CHECK: llvm.return
    llvm.return
  }

  // CHECK-LABEL: llvm.func @reuse_tests
  // CHECK-SAME: (%[[THREAD_LIMIT:.*]]: i32)
  llvm.func @reuse_tests() {
    // CHECK-NEXT: %[[CONST_THREAD_LIMIT:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[GLOBAL:.*]] = llvm.mlir.addressof @global_scalar : !llvm.ptr
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.addressof @global_scalar : !llvm.ptr
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target_data map_entries(%2 : !llvm.ptr) {
      // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[GLOBAL]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
      // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[GLOBAL]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
      // CHECK-NEXT: %[[MAP2:.*]] = omp.map.info var_ptr(%[[GLOBAL]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
      %5 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
      %6 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr

      // CHECK-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP0]] -> %{{.*}}, %[[MAP1]] -> %{{.*}} : !llvm.ptr, !llvm.ptr)
      omp.target kernel_type(generic) map_entries(%5 -> %arg0, %6 -> %arg1 : !llvm.ptr, !llvm.ptr) {
        omp.terminator
      }
      omp.terminator
    }

    // CHECK-NOT: llvm.load
    // CHECK-NOT: omp.map.info
    %3 = llvm.load %1 : !llvm.ptr -> i32
    %4 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr

    // CHECK: omp.target kernel_type(generic) thread_limit(%[[THREAD_LIMIT]] : i32) map_entries(%[[MAP2]] -> %{{.*}} : !llvm.ptr)
    omp.target kernel_type(generic) thread_limit(%3 : i32) map_entries(%4 -> %arg0 : !llvm.ptr) {
      omp.terminator
    }

    // CHECK: omp.target kernel_type(generic) thread_limit(%[[CONST_THREAD_LIMIT]] : i32)
    omp.target kernel_type(generic) thread_limit(%0 : i32) {
      omp.terminator
    }

    // CHECK: omp.target kernel_type(generic) thread_limit(%[[CONST_THREAD_LIMIT]] : i32)
    omp.target kernel_type(generic) thread_limit(%0 : i32) {
      omp.terminator
    }

    // CHECK: llvm.return
    llvm.return
  }

  // CHECK-LABEL: llvm.func @all_non_map_clauses
  // CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i1)
  llvm.func @all_non_map_clauses(%arg0: !llvm.ptr, %arg1: i32, %arg2: i1) {
    %0 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target_data device(%arg1 : i32) if(%arg2) map_entries(%0 : !llvm.ptr) {
      omp.terminator
    }

    // CHECK-NEXT: omp.target kernel_type(generic) allocate(%[[ARG0]] : !llvm.ptr -> %[[ARG0]] : !llvm.ptr) thread_limit(%[[ARG1]] : i32) private(@privatizer %[[ARG0]] -> %{{.*}} : !llvm.ptr)
    omp.target kernel_type(generic) allocate(%arg0 : !llvm.ptr -> %arg0 : !llvm.ptr) depend(taskdependin -> %arg0 : !llvm.ptr) device(%arg1 : i32) if(%arg2) thread_limit(%arg1 : i32) in_reduction(@reduction %arg0 -> %arg3 : !llvm.ptr) private(@privatizer %arg0 -> %arg4 : !llvm.ptr) {
      omp.terminator
    }

    // CHECK-NOT: omp.target_enter_data
    // CHECK-NOT: omp.target_exit_data
    // CHECK-NOT: omp.target_update
    omp.target_enter_data depend(taskdependin -> %arg0 : !llvm.ptr) device(%arg1 : i32) if(%arg2)
    omp.target_exit_data depend(taskdependin -> %arg0 : !llvm.ptr) device(%arg1 : i32) if(%arg2)
    omp.target_update depend(taskdependin -> %arg0 : !llvm.ptr) device(%arg1 : i32) if(%arg2)

    // CHECK: llvm.return
    llvm.return
  }

  // CHECK-LABEL: llvm.func @private_with_map_idx
  // CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: i64)
  llvm.func @private_with_map_idx(%arg0: !llvm.ptr, %arg1: i64) {
    // CHECK-NEXT: %[[VAL0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    // CHECK-NEXT: %[[VAL1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    // CHECK-NEXT: %[[VAL2:.*]] = llvm.insertvalue %[[ARG0]], %[[VAL1]][0] : !llvm.struct<(ptr, i64)>
    // CHECK-NEXT: %[[VAL3:.*]] = llvm.insertvalue %[[ARG1]], %[[VAL2]][1] : !llvm.struct<(ptr, i64)>
    // CHECK-NEXT: %[[VAL4:.*]] = llvm.extractvalue %[[VAL3]][0] : !llvm.struct<(ptr, i64)>
    // CHECK-NEXT: %[[VAL5:.*]] = llvm.insertvalue %[[VAL4]], %[[VAL0]][0] : !llvm.struct<(ptr, i64)>
    // CHECK-NEXT: %[[VAL6:.*]] = llvm.extractvalue %[[VAL3]][1] : !llvm.struct<(ptr, i64)>
    // CHECK-NEXT: %[[VAL7:.*]] = llvm.insertvalue %[[VAL6]], %[[VAL5]][1] : !llvm.struct<(ptr, i64)>
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, i64)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, i64)>
    %3 = llvm.extractvalue %2[0] : !llvm.struct<(ptr, i64)>
    %4 = llvm.extractvalue %2[1] : !llvm.struct<(ptr, i64)>
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %6 = llvm.insertvalue %3, %5[0] : !llvm.struct<(ptr, i64)>
    %7 = llvm.insertvalue %4, %6[1] : !llvm.struct<(ptr, i64)>

    // CHECK: omp.target kernel_type(generic) private(@struct_firstprivatizer %[[VAL7]] -> %{{.*}} [map_idx=0] : !llvm.struct<(ptr, i64)>)
    omp.target kernel_type(generic) private(@struct_firstprivatizer %7 -> %arg2 [map_idx=0] : !llvm.struct<(ptr, i64)>) {
      omp.terminator
    }

    // CHECK: llvm.return
    llvm.return
  }

  omp.private {type = firstprivate} @struct_firstprivatizer : !llvm.struct<(ptr, i64)> copy {
  ^bb0(%arg0: !llvm.struct<(ptr, i64)>, %arg1: !llvm.struct<(ptr, i64)>):
    omp.yield(%arg0 : !llvm.struct<(ptr, i64)>)
  }

  // CHECK-LABEL: llvm.func @map_from_global_gep
  // CHECK-SAME: (%[[ARG0:.*]]: i64)
  llvm.func @map_from_global_gep(%arg0: i64) {
    // CHECK-NEXT: %[[GLOBAL:.*]] = llvm.mlir.addressof @global_array : !llvm.ptr
    // CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[GLOBAL]][%[[ARG0]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %0 = llvm.mlir.addressof @global_array : !llvm.ptr
    %1 = llvm.getelementptr %0[%arg0] : (!llvm.ptr, i64) -> !llvm.ptr, i8

    // CHECK-NEXT: %[[MAP:.*]] = omp.map.info var_ptr(%[[GEP]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr

    // CHECK-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP]] -> %{{.*}} : !llvm.ptr)
    omp.target kernel_type(generic) map_entries(%2 -> %arg1 : !llvm.ptr) {
      omp.terminator
    }

    // CHECK: llvm.return
    llvm.return
  }

  // CHECK-LABEL: llvm.func @reciprocal_a
  // CHECK-SAME: (%[[PLACEHOLDER:.*]]: !llvm.ptr)
  llvm.func @reciprocal_a() {
    // CHECK-NEXT: omp.map.info
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr

    // CHECK-NEXT: omp.target
    omp.target kernel_type(generic) map_entries(%2 -> %arg0 : !llvm.ptr) {
      omp.terminator
    }

    // CHECK-NOT: @reciprocal_b
    llvm.call @reciprocal_b(%0) : (i64) -> ()

    // CHECK: llvm.return
    llvm.return
  }

  // CHECK-LABEL: llvm.func @reciprocal_b
  // CHECK-SAME: (%[[ARG0:.*]]: i64, %[[PLACEHOLDER:.*]]: !llvm.ptr)
  llvm.func @reciprocal_b(%arg0 : i64) {
    // CHECK-NEXT: omp.map.info
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr

    // CHECK-NEXT: omp.target
    omp.target kernel_type(generic) map_entries(%2 -> %arg1 : !llvm.ptr) {
      omp.terminator
    }

    // CHECK-NOT: @reciprocal_a
    llvm.call @reciprocal_a() : () -> ()

    // CHECK: llvm.return
    llvm.return
  }

  llvm.func @foo() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>, sym_visibility = "private"}
  omp.private {type = firstprivate} @privatizer : i32 copy {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.load %arg0 : !llvm.ptr -> i32
    llvm.store %0, %arg1 : i32, !llvm.ptr
    omp.yield(%arg1 : !llvm.ptr)
  }
  omp.declare_reduction @reduction : i32 init {

  ^bb0(%arg0: i32):
    %0 = llvm.mlir.constant(0 : i32) : i32
    omp.yield(%0 : i32)
  } combiner {

  ^bb0(%arg0: i32, %arg1: i32):
    %0 = llvm.add %arg0, %arg1 : i32
    omp.yield(%0 : i32)
  }

  // CHECK: llvm.mlir.global internal constant @global_scalar
  // CHECK: llvm.mlir.global internal @global_array
  llvm.mlir.global external constant @global_scalar() {addr_space = 0 : i32} : i32
  llvm.mlir.global external @global_array() {addr_space = 0 : i32} : !llvm.array<8 x i8>

  // CHECK: llvm.mlir.global external @declare_target_enter_any
  // CHECK: llvm.mlir.global external @declare_target_enter_host
  // CHECK: llvm.mlir.global external @declare_target_enter_nohost
  llvm.mlir.global external @declare_target_enter_any() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false>} : i32
  llvm.mlir.global external @declare_target_enter_host() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter), automap = false>} : i32
  llvm.mlir.global external @declare_target_enter_nohost() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter), automap = false>} : i32

  // CHECK: llvm.mlir.global external @declare_target_link_any
  // CHECK: llvm.mlir.global external @declare_target_link_host
  // CHECK: llvm.mlir.global external @declare_target_link_nohost
  llvm.mlir.global external @declare_target_link_any() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link), automap = false>} : i32
  llvm.mlir.global external @declare_target_link_host() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (link), automap = false>} : i32
  llvm.mlir.global external @declare_target_link_nohost() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (link), automap = false>} : i32

  llvm.func @LangRTPlaceholderFunc(!llvm.ptr {llvm.nocapture}, !llvm.ptr {llvm.nocapture}, !llvm.ptr, i8 {llvm.signext}, i32) attributes {sym_visibility = "private"}
}

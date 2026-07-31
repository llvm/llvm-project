// RUN: mlir-opt %s -acc-to-llvm -split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @test_init_shutdown
// CHECK-NOT: acc.init
// CHECK-NOT: acc.shutdown
// The device type is materialized first, then the ident, the flags and finally
// the device number widened to i64. With the default ACCRuntimeCallConfig
// mapping (dialect ordinal), nvidia is 5.
// CHECK: %[[NVIDIA:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK: %[[IDENT:.*]] = llvm.mlir.addressof @[[ID:ident_[^ ]+]]
// CHECK: %[[FLAGS:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[DEVNUM:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: llvm.call @__tgt_acc_init(%[[IDENT]], %[[FLAGS]], %[[NVIDIA]], %[[DEVNUM]]) : (!llvm.ptr, i64, i64, i64) -> ()
// CHECK: llvm.call @__tgt_acc_shutdown
// Without a device number, the current device (-1) is used, once per device type.
// CHECK: %[[NVIDIA2:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK: %[[CURRENT:.*]] = llvm.mlir.constant(-1 : i64) : i64
// CHECK: llvm.call @__tgt_acc_init(%{{.*}}, %{{.*}}, %[[NVIDIA2]], %[[CURRENT]])
// With the default mapping, host is 3.
// CHECK: %[[HOST:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK: llvm.call @__tgt_acc_init(%{{.*}}, %{{.*}}, %[[HOST]], %{{.*}})
// CHECK: llvm.call @__tgt_acc_shutdown
// CHECK: llvm.call @__tgt_acc_shutdown
// Without device types, DeviceType::None maps to 0 under the default mapping.
// CHECK: %[[NONE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @__tgt_acc_init(%{{.*}}, %{{.*}}, %[[NONE]], %{{.*}})
// CHECK: llvm.call @__tgt_acc_shutdown
// CHECK: llvm.call @__tgt_acc_init
// CHECK: llvm.call @__tgt_acc_shutdown

module {
  func.func @test_init_shutdown() {
    %c2_i32 = arith.constant 2 : i32
    acc.init device_num(%c2_i32 : i32) attributes {device_types = [#acc.device_type<nvidia>]}
    acc.shutdown device_num(%c2_i32 : i32) attributes {device_types = [#acc.device_type<nvidia>]}
    acc.init attributes {device_types = [#acc.device_type<nvidia>, #acc.device_type<host>]}
    acc.shutdown attributes {device_types = [#acc.device_type<nvidia>, #acc.device_type<host>]}
    acc.init device_num(%c2_i32 : i32)
    acc.shutdown device_num(%c2_i32 : i32)
    acc.init
    acc.shutdown
    return
  }
}

// -----

// CHECK-LABEL: llvm.func @test_set
// CHECK: llvm.call @__tgt_acc_set_default_async
// CHECK: %[[NVIDIA:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK: llvm.call @__tgt_acc_set_device_num(%{{.*}}, %{{.*}}, %[[NVIDIA]], %{{.*}})

module {
  func.func @test_set() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    acc.set default_async(%c1_i32 : i32) device_num(%c0_i32 : i32) attributes {device_type = #acc.device_type<nvidia>}
    return
  }
}

// -----

// CHECK-LABEL: llvm.func @test_set_device_type
// CHECK: %[[HOST:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK: llvm.call @__tgt_acc_set_device_type(%{{.*}}, %{{.*}}, %[[HOST]])

module {
  func.func @test_set_device_type() {
    acc.set attributes {device_type = #acc.device_type<host>}
    return
  }
}

// -----

// CHECK-LABEL: llvm.func @test_if
// CHECK: llvm.cond_br %{{.*}}, ^[[THEN:bb[0-9]+]], ^[[CONT:bb[0-9]+]]
// CHECK: ^[[THEN]]:
// CHECK: llvm.call @__tgt_acc_init
// CHECK: llvm.br ^[[CONT]]
// CHECK: ^[[CONT]]:

module {
  func.func @test_if(%cond: i1) {
    acc.init if(%cond) attributes {device_types = [#acc.device_type<nvidia>]}
    return
  }
}

// -----

// The ident is a constant global whose source field points at the location
// string. Call sites take the address of that ident, not of the string.

// CHECK: llvm.mlir.global internal constant @[[$SRC:loc_10_1_[0-9]+]](";init-shutdown-set.mlir;test_init_with_loc;10;1;;\00")
// CHECK: llvm.mlir.global internal constant @[[$IDENT:ident_loc_10_1_[0-9]+]]() {{.*}} : !llvm.struct<(i32, i32, i32, i32, ptr)> {
// CHECK: llvm.mlir.zero : !llvm.struct<(i32, i32, i32, i32, ptr)>
// CHECK: llvm.mlir.addressof @[[$SRC]]
// CHECK: llvm.getelementptr
// CHECK: llvm.insertvalue {{.*}}[4]
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @test_init_with_loc
// CHECK: llvm.mlir.addressof @[[$IDENT]]
// CHECK: llvm.call @__tgt_acc_init

#loc = loc("init-shutdown-set.mlir":10:1)
module {
  func.func @test_init_with_loc() {
    acc.init loc(#loc)
    return
  }
}

// -----

// Operations without file:line information fall back to an unknown ident.

// CHECK: llvm.mlir.global internal constant @loc__(";unknown;unknown;0;0;;\00")
// CHECK: llvm.mlir.global internal constant @ident_loc__() {{.*}} : !llvm.struct<(i32, i32, i32, i32, ptr)> {
// CHECK: llvm.mlir.zero : !llvm.struct<(i32, i32, i32, i32, ptr)>
// CHECK: llvm.mlir.addressof @loc__
// CHECK: llvm.getelementptr
// CHECK: llvm.insertvalue {{.*}}[4]
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @test_init_unknown_loc
// CHECK: llvm.mlir.addressof @ident_loc__
// CHECK: llvm.call @__tgt_acc_init

module {
  func.func @test_init_unknown_loc() {
    acc.init loc(unknown)
    return
  }
}

// -----

// Two operations at the same line and column of different files must not share
// a location global, otherwise the ident would name the wrong file.

// CHECK-DAG: llvm.mlir.global internal constant @[[$LOC_A:loc_7_3_[0-9]+]](";a.mlir;test_distinct_files;7;3;;\00")
// CHECK-DAG: llvm.mlir.global internal constant @[[$LOC_B:loc_7_3_[0-9]+]](";b.mlir;test_distinct_files;7;3;;\00")
// CHECK-LABEL: llvm.func @test_distinct_files
// CHECK: llvm.mlir.addressof @ident_[[$LOC_A]]
// CHECK: llvm.call @__tgt_acc_init
// CHECK: llvm.mlir.addressof @ident_[[$LOC_B]]
// CHECK: llvm.call @__tgt_acc_shutdown

module {
  func.func @test_distinct_files() {
    acc.init loc("a.mlir":7:3)
    acc.shutdown loc("b.mlir":7:3)
    return
  }
}

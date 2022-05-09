// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @ops
// CHECK-SAME: (%[[I32:.*]]: i32, %[[FLOAT:.*]]: f32, %[[I8PTR1:.*]]: !llvm.ptr<i8>, %[[I8PTR2:.*]]: !llvm.ptr<i8>, %[[BOOL:.*]]: i1, %[[VI8PTR1:.*]]: !llvm.vec<2 x ptr<i8>>)
func.func @ops(%arg0: i32, %arg1: f32,
          %arg2: !llvm.ptr<i8>, %arg3: !llvm.ptr<i8>,
          %arg4: i1, %arg5 : !llvm.vec<2x!llvm.ptr<i8>>) {
// Integer arithmetic binary operations.
//
// CHECK: {{.*}} = llvm.add %[[I32]], %[[I32]] : i32
// CHECK: {{.*}} = llvm.sub %[[I32]], %[[I32]] : i32
// CHECK: {{.*}} = llvm.mul %[[I32]], %[[I32]] : i32
// CHECK: {{.*}} = llvm.udiv %[[I32]], %[[I32]] : i32
// CHECK: {{.*}} = llvm.sdiv %[[I32]], %[[I32]] : i32
// CHECK: {{.*}} = llvm.urem %[[I32]], %[[I32]] : i32
// CHECK: {{.*}} = llvm.srem %[[I32]], %[[I32]] : i32
// CHECK: {{.*}} = llvm.icmp "ne" %[[I32]], %[[I32]] : i32
// CHECK: {{.*}} = llvm.icmp "ne" %[[I8PTR1]], %[[I8PTR1]] : !llvm.ptr<i8>
// CHECK: {{.*}} = llvm.icmp "ne" %[[VI8PTR1]], %[[VI8PTR1]] : !llvm.vec<2 x ptr<i8>>
  %0 = llvm.add %arg0, %arg0 : i32
  %1 = llvm.sub %arg0, %arg0 : i32
  %2 = llvm.mul %arg0, %arg0 : i32
  %3 = llvm.udiv %arg0, %arg0 : i32
  %4 = llvm.sdiv %arg0, %arg0 : i32
  %5 = llvm.urem %arg0, %arg0 : i32
  %6 = llvm.srem %arg0, %arg0 : i32
  %7 = llvm.icmp "ne" %arg0, %arg0 : i32
  %ptrcmp = llvm.icmp "ne" %arg2, %arg2 : !llvm.ptr<i8>
  %vptrcmp = llvm.icmp "ne" %arg5, %arg5 : !llvm.vec<2 x ptr<i8>>

// Floating point binary operations.
//
// CHECK: {{.*}} = llvm.fadd %[[FLOAT]], %[[FLOAT]] : f32
// CHECK: {{.*}} = llvm.fsub %[[FLOAT]], %[[FLOAT]] : f32
// CHECK: {{.*}} = llvm.fmul %[[FLOAT]], %[[FLOAT]] : f32
// CHECK: {{.*}} = llvm.fdiv %[[FLOAT]], %[[FLOAT]] : f32
// CHECK: {{.*}} = llvm.frem %[[FLOAT]], %[[FLOAT]] : f32
  %8 = llvm.fadd %arg1, %arg1 : f32
  %9 = llvm.fsub %arg1, %arg1 : f32
  %10 = llvm.fmul %arg1, %arg1 : f32
  %11 = llvm.fdiv %arg1, %arg1 : f32
  %12 = llvm.frem %arg1, %arg1 : f32

// Memory-related operations.
//
// CHECK-NEXT:  %[[ALLOCA:.*]] = llvm.alloca %[[I32]] x f64 : (i32) -> !llvm.ptr<f64>
// CHECK-NEXT:  %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][%[[I32]], %[[I32]]] : (!llvm.ptr<f64>, i32, i32) -> !llvm.ptr<f64>
// CHECK-NEXT:  %[[VALUE:.*]] = llvm.load %[[GEP]] : !llvm.ptr<f64>
// CHECK-NEXT:  llvm.store %[[VALUE]], %[[ALLOCA]] : !llvm.ptr<f64>
// CHECK-NEXT:  %{{.*}} = llvm.bitcast %[[ALLOCA]] : !llvm.ptr<f64> to !llvm.ptr<i64>
  %13 = llvm.alloca %arg0 x f64 : (i32) -> !llvm.ptr<f64>
  %14 = llvm.getelementptr %13[%arg0, %arg0] : (!llvm.ptr<f64>, i32, i32) -> !llvm.ptr<f64>
  %15 = llvm.load %14 : !llvm.ptr<f64>
  llvm.store %15, %13 : !llvm.ptr<f64>
  %16 = llvm.bitcast %13 : !llvm.ptr<f64> to !llvm.ptr<i64>

// Function call-related operations.
//
// CHECK: %[[STRUCT:.*]] = llvm.call @foo(%[[I32]]) : (i32) -> !llvm.struct<(i32, f64, i32)>
// CHECK: %[[VALUE:.*]] = llvm.extractvalue %[[STRUCT]][0] : !llvm.struct<(i32, f64, i32)>
// CHECK: %[[NEW_STRUCT:.*]] = llvm.insertvalue %[[VALUE]], %[[STRUCT]][2] : !llvm.struct<(i32, f64, i32)>
// CHECK: %[[FUNC:.*]] = llvm.mlir.addressof @foo : !llvm.ptr<func<struct<(i32, f64, i32)> (i32)>>
// CHECK: %{{.*}} = llvm.call %[[FUNC]](%[[I32]]) : (i32) -> !llvm.struct<(i32, f64, i32)>
  %17 = llvm.call @foo(%arg0) : (i32) -> !llvm.struct<(i32, f64, i32)>
  %18 = llvm.extractvalue %17[0] : !llvm.struct<(i32, f64, i32)>
  %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(i32, f64, i32)>
  %20 = llvm.mlir.addressof @foo : !llvm.ptr<func<struct<(i32, f64, i32)> (i32)>>
  %21 = llvm.call %20(%arg0) : (i32) -> !llvm.struct<(i32, f64, i32)>


// Terminator operations and their successors.
//
// CHECK: llvm.br ^[[BB1:.*]]
  llvm.br ^bb1

// CHECK: ^[[BB1]]
^bb1:
// CHECK: llvm.cond_br %7, ^[[BB2:.*]], ^[[BB3:.*]]
  llvm.cond_br %7, ^bb2, ^bb3

// CHECK: ^[[BB2]]
^bb2:
// CHECK: %{{.*}} = llvm.mlir.undef : !llvm.struct<(i32, f64, i32)>
// CHECK: %{{.*}} = llvm.mlir.constant(42 : i64) : i47
  %22 = llvm.mlir.undef : !llvm.struct<(i32, f64, i32)>
  %23 = llvm.mlir.constant(42) : i47
  // CHECK:      llvm.switch %0 : i32, ^[[BB3]] [
  // CHECK-NEXT:   1: ^[[BB4:.*]],
  // CHECK-NEXT:   2: ^[[BB5:.*]],
  // CHECK-NEXT:   3: ^[[BB6:.*]]
  // CHECK-NEXT: ]
  llvm.switch %0 : i32, ^bb3 [
    1: ^bb4,
    2: ^bb5,
    3: ^bb6
  ]

// CHECK: ^[[BB3]]
^bb3:
// CHECK:      llvm.switch %0 : i32, ^[[BB7:.*]] [
// CHECK-NEXT: ]
  llvm.switch %0 : i32, ^bb7 [
  ]

// CHECK: ^[[BB4]]
^bb4:
  llvm.switch %0 : i32, ^bb7 [
  ]

// CHECK: ^[[BB5]]
^bb5:
  llvm.switch %0 : i32, ^bb7 [
  ]

// CHECK: ^[[BB6]]
^bb6:
  llvm.switch %0 : i32, ^bb7 [
  ]

// CHECK: ^[[BB7]]
^bb7:
// Misc operations.
// CHECK: %{{.*}} = llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, i32
  %24 = llvm.select %7, %0, %1 : i1, i32

// Integer to pointer and pointer to integer conversions.
//
// CHECK: %[[PTR:.*]] = llvm.inttoptr %[[I32]] : i32 to !llvm.ptr<i32>
// CHECK: %{{.*}} = llvm.ptrtoint %[[PTR]] : !llvm.ptr<i32> to i32
  %25 = llvm.inttoptr %arg0 : i32 to !llvm.ptr<i32>
  %26 = llvm.ptrtoint %25 : !llvm.ptr<i32> to i32

// Extended and Quad floating point
//
// CHECK: %{{.*}} = llvm.fpext %[[FLOAT]] : f32 to f80
// CHECK: %{{.*}} = llvm.fpext %[[FLOAT]] : f32 to f128
  %27 = llvm.fpext %arg1 : f32 to f80
  %28 = llvm.fpext %arg1 : f32 to f128

// CHECK: %{{.*}} = llvm.fneg %[[FLOAT]] : f32
  %29 = llvm.fneg %arg1 : f32

// CHECK: "llvm.intr.sin"(%[[FLOAT]]) : (f32) -> f32
  %30 = "llvm.intr.sin"(%arg1) : (f32) -> f32

// CHECK: "llvm.intr.pow"(%[[FLOAT]], %[[FLOAT]]) : (f32, f32) -> f32
  %31 = "llvm.intr.pow"(%arg1, %arg1) : (f32, f32) -> f32

// CHECK: "llvm.intr.powi"(%[[FLOAT]], %[[I32]]) : (f32, i32) -> f32
  %a31 = "llvm.intr.powi"(%arg1, %arg0) : (f32, i32) -> f32

// CHECK: "llvm.intr.bitreverse"(%{{.*}}) : (i32) -> i32
  %32 = "llvm.intr.bitreverse"(%arg0) : (i32) -> i32

// CHECK: "llvm.intr.ctpop"(%{{.*}}) : (i32) -> i32
  %33 = "llvm.intr.ctpop"(%arg0) : (i32) -> i32

// CHECK: "llvm.intr.memcpy"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i32, i1) -> ()
  "llvm.intr.memcpy"(%arg2, %arg3, %arg0, %arg4) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i32, i1) -> ()

// CHECK: "llvm.intr.memcpy"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i32, i1) -> ()
  "llvm.intr.memcpy"(%arg2, %arg3, %arg0, %arg4) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i32, i1) -> ()

// CHECK: %[[SZ:.*]] = llvm.mlir.constant
  %sz = llvm.mlir.constant(10: i64) : i64
// CHECK: "llvm.intr.memcpy.inline"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "llvm.intr.memcpy.inline"(%arg2, %arg3, %sz, %arg4) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()

// CHECK:  llvm.return
  llvm.return
}

// CHECK-LABEL: @gep
llvm.func @gep(%ptr: !llvm.ptr<struct<(i32, struct<(i32, f32)>)>>, %idx: i64,
               %ptr2: !llvm.ptr<struct<(array<10xf32>)>>) {
  // CHECK: llvm.getelementptr %{{.*}}[%{{.*}}, 1, 0] : (!llvm.ptr<struct<(i32, struct<(i32, f32)>)>>, i64) -> !llvm.ptr<i32>
  llvm.getelementptr %ptr[%idx, 1, 0] : (!llvm.ptr<struct<(i32, struct<(i32, f32)>)>>, i64) -> !llvm.ptr<i32>
  // CHECK: llvm.getelementptr %{{.*}}[%{{.*}}, 0, %{{.*}}] : (!llvm.ptr<struct<(array<10 x f32>)>>, i64, i64) -> !llvm.ptr<f32>
  llvm.getelementptr %ptr2[%idx, 0, %idx] : (!llvm.ptr<struct<(array<10 x f32>)>>, i64, i64) -> !llvm.ptr<f32>
  llvm.return
}

// An larger self-contained function.
// CHECK-LABEL: llvm.func @foo(%{{.*}}: i32) -> !llvm.struct<(i32, f64, i32)> {
llvm.func @foo(%arg0: i32) -> !llvm.struct<(i32, f64, i32)> {
// CHECK:  %[[V0:.*]] = llvm.mlir.constant(3 : i64) : i32
// CHECK:  %[[V1:.*]] = llvm.mlir.constant(3 : i64) : i32
// CHECK:  %[[V2:.*]] = llvm.mlir.constant(4.200000e+01 : f64) : f64
// CHECK:  %[[V3:.*]] = llvm.mlir.constant(4.200000e+01 : f64) : f64
// CHECK:  %[[V4:.*]] = llvm.add %[[V0]], %[[V1]] : i32
// CHECK:  %[[V5:.*]] = llvm.mul %[[V4]], %[[V1]] : i32
// CHECK:  %[[V6:.*]] = llvm.fadd %[[V2]], %[[V3]] : f64
// CHECK:  %[[V7:.*]] = llvm.fsub %[[V3]], %[[V6]] : f64
// CHECK:  %[[V8:.*]] = llvm.mlir.constant(1 : i64) : i1
// CHECK:  llvm.cond_br %[[V8]], ^[[BB1:.*]](%[[V4]] : i32), ^[[BB2:.*]](%[[V4]] : i32)
  %0 = llvm.mlir.constant(3) : i32
  %1 = llvm.mlir.constant(3) : i32
  %2 = llvm.mlir.constant(4.200000e+01) : f64
  %3 = llvm.mlir.constant(4.200000e+01) : f64
  %4 = llvm.add %0, %1 : i32
  %5 = llvm.mul %4, %1 : i32
  %6 = llvm.fadd %2, %3 : f64
  %7 = llvm.fsub %3, %6 : f64
  %8 = llvm.mlir.constant(1) : i1
  llvm.cond_br %8, ^bb1(%4 : i32), ^bb2(%4 : i32)

// CHECK:^[[BB1]](%[[V9:.*]]: i32):
// CHECK:  %[[V10:.*]] = llvm.call @foo(%[[V9]]) : (i32) -> !llvm.struct<(i32, f64, i32)>
// CHECK:  %[[V11:.*]] = llvm.extractvalue %[[V10]][0] : !llvm.struct<(i32, f64, i32)>
// CHECK:  %[[V12:.*]] = llvm.extractvalue %[[V10]][1] : !llvm.struct<(i32, f64, i32)>
// CHECK:  %[[V13:.*]] = llvm.extractvalue %[[V10]][2] : !llvm.struct<(i32, f64, i32)>
// CHECK:  %[[V14:.*]] = llvm.mlir.undef : !llvm.struct<(i32, f64, i32)>
// CHECK:  %[[V15:.*]] = llvm.insertvalue %[[V5]], %[[V14]][0] : !llvm.struct<(i32, f64, i32)>
// CHECK:  %[[V16:.*]] = llvm.insertvalue %[[V7]], %[[V15]][1] : !llvm.struct<(i32, f64, i32)>
// CHECK:  %[[V17:.*]] = llvm.insertvalue %[[V11]], %[[V16]][2] : !llvm.struct<(i32, f64, i32)>
// CHECK:  llvm.return %[[V17]] : !llvm.struct<(i32, f64, i32)>
^bb1(%9: i32):
  %10 = llvm.call @foo(%9) : (i32) -> !llvm.struct<(i32, f64, i32)>
  %11 = llvm.extractvalue %10[0] : !llvm.struct<(i32, f64, i32)>
  %12 = llvm.extractvalue %10[1] : !llvm.struct<(i32, f64, i32)>
  %13 = llvm.extractvalue %10[2] : !llvm.struct<(i32, f64, i32)>
  %14 = llvm.mlir.undef : !llvm.struct<(i32, f64, i32)>
  %15 = llvm.insertvalue %5, %14[0] : !llvm.struct<(i32, f64, i32)>
  %16 = llvm.insertvalue %7, %15[1] : !llvm.struct<(i32, f64, i32)>
  %17 = llvm.insertvalue %11, %16[2] : !llvm.struct<(i32, f64, i32)>
  llvm.return %17 : !llvm.struct<(i32, f64, i32)>

// CHECK:^[[BB2]](%[[V18:.*]]: i32):
// CHECK:  %[[V19:.*]] = llvm.mlir.undef : !llvm.struct<(i32, f64, i32)>
// CHECK:  %[[V20:.*]] = llvm.insertvalue %[[V18]], %[[V19]][0] : !llvm.struct<(i32, f64, i32)>
// CHECK:  %[[V21:.*]] = llvm.insertvalue %[[V7]], %[[V20]][1] : !llvm.struct<(i32, f64, i32)>
// CHECK:  %[[V22:.*]] = llvm.insertvalue %[[V5]], %[[V21]][2] : !llvm.struct<(i32, f64, i32)>
// CHECK:  llvm.return %[[V22]] : !llvm.struct<(i32, f64, i32)>
^bb2(%18: i32):
  %19 = llvm.mlir.undef : !llvm.struct<(i32, f64, i32)>
  %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(i32, f64, i32)>
  %21 = llvm.insertvalue %7, %20[1] : !llvm.struct<(i32, f64, i32)>
  %22 = llvm.insertvalue %5, %21[2] : !llvm.struct<(i32, f64, i32)>
  llvm.return %22 : !llvm.struct<(i32, f64, i32)>
}

// CHECK-LABEL: @casts
// CHECK-SAME: (%[[I32:.*]]: i32, %[[I64:.*]]: i64, %[[V4I32:.*]]: vector<4xi32>, %[[V4I64:.*]]: vector<4xi64>, %[[I32PTR:.*]]: !llvm.ptr<i32>)
func.func @casts(%arg0: i32, %arg1: i64, %arg2: vector<4xi32>,
            %arg3: vector<4xi64>, %arg4: !llvm.ptr<i32>) {
// CHECK:  = llvm.sext %[[I32]] : i32 to i56
  %0 = llvm.sext %arg0 : i32 to i56
// CHECK:  = llvm.zext %[[I32]] : i32 to i64
  %1 = llvm.zext %arg0 : i32 to i64
// CHECK:  = llvm.trunc %[[I64]] : i64 to i56
  %2 = llvm.trunc %arg1 : i64 to i56
// CHECK:  = llvm.sext %[[V4I32]] : vector<4xi32> to vector<4xi56>
  %3 = llvm.sext %arg2 : vector<4xi32> to vector<4xi56>
// CHECK:  = llvm.zext %[[V4I32]] : vector<4xi32> to vector<4xi64>
  %4 = llvm.zext %arg2 : vector<4xi32> to vector<4xi64>
// CHECK:  = llvm.trunc %[[V4I64]] : vector<4xi64> to vector<4xi56>
  %5 = llvm.trunc %arg3 : vector<4xi64> to vector<4xi56>
// CHECK:  = llvm.sitofp %[[I32]] : i32 to f32
  %6 = llvm.sitofp %arg0 : i32 to f32
// CHECK: %[[FLOAT:.*]] = llvm.uitofp %[[I32]] : i32 to f32
  %7 = llvm.uitofp %arg0 : i32 to f32
// CHECK:  = llvm.fptosi %[[FLOAT]] : f32 to i32
  %8 = llvm.fptosi %7 : f32 to i32
// CHECK:  = llvm.fptoui %[[FLOAT]] : f32 to i32
  %9 = llvm.fptoui %7 : f32 to i32
// CHECK:  = llvm.addrspacecast %[[I32PTR]] : !llvm.ptr<i32> to !llvm.ptr<i32, 2>
  %10 = llvm.addrspacecast %arg4 : !llvm.ptr<i32> to !llvm.ptr<i32, 2>
  llvm.return
}

// CHECK-LABEL: @vect
func.func @vect(%arg0: vector<4xf32>, %arg1: i32, %arg2: f32) {
// CHECK:  = llvm.extractelement {{.*}} : vector<4xf32>
  %0 = llvm.extractelement %arg0[%arg1 : i32] : vector<4xf32>
// CHECK:  = llvm.insertelement {{.*}} : vector<4xf32>
  %1 = llvm.insertelement %arg2, %arg0[%arg1 : i32] : vector<4xf32>
// CHECK:  = llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32, 7 : i32] : vector<4xf32>, vector<4xf32>
  %2 = llvm.shufflevector %arg0, %arg0 [0 : i32, 0 : i32, 0 : i32, 0 : i32, 7 : i32] : vector<4xf32>, vector<4xf32>
// CHECK:  = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : vector<4xf32>
  %3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
  return
}

// CHECK-LABEL: @scalable_vect
func.func @scalable_vect(%arg0: vector<[4]xf32>, %arg1: i32, %arg2: f32) {
// CHECK:  = llvm.extractelement {{.*}} : vector<[4]xf32>
  %0 = llvm.extractelement %arg0[%arg1 : i32] : vector<[4]xf32>
// CHECK:  = llvm.insertelement {{.*}} : vector<[4]xf32>
  %1 = llvm.insertelement %arg2, %arg0[%arg1 : i32] : vector<[4]xf32>
// CHECK:  = llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<[4]xf32>, vector<[4]xf32>
  %2 = llvm.shufflevector %arg0, %arg0 [0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<[4]xf32>, vector<[4]xf32>
// CHECK:  = llvm.mlir.constant(dense<1.000000e+00> : vector<[4]xf32>) : vector<[4]xf32>
  %3 = llvm.mlir.constant(dense<1.0> : vector<[4]xf32>) : vector<[4]xf32>
  return
}

// CHECK-LABEL: @alloca
func.func @alloca(%size : i64) {
  // CHECK: llvm.alloca %{{.*}} x i32 : (i64) -> !llvm.ptr<i32>
  llvm.alloca %size x i32 {alignment = 0} : (i64) -> (!llvm.ptr<i32>)
  // CHECK: llvm.alloca %{{.*}} x i32 {alignment = 8 : i64} : (i64) -> !llvm.ptr<i32>
  llvm.alloca %size x i32 {alignment = 8} : (i64) -> (!llvm.ptr<i32>)
  llvm.return
}

// CHECK-LABEL: @null
func.func @null() {
  // CHECK: llvm.mlir.null : !llvm.ptr<i8>
  %0 = llvm.mlir.null : !llvm.ptr<i8>
  // CHECK: llvm.mlir.null : !llvm.ptr<struct<(ptr<func<void (i32, ptr<func<void ()>>)>>, i64)>>
  %1 = llvm.mlir.null : !llvm.ptr<struct<(ptr<func<void (i32, ptr<func<void ()>>)>>, i64)>>
  llvm.return
}

// CHECK-LABEL: @atomicrmw
func.func @atomicrmw(%ptr : !llvm.ptr<f32>, %val : f32) {
  // CHECK: llvm.atomicrmw fadd %{{.*}}, %{{.*}} monotonic : f32
  %0 = llvm.atomicrmw fadd %ptr, %val monotonic : f32
  llvm.return
}

// CHECK-LABEL: @cmpxchg
func.func @cmpxchg(%ptr : !llvm.ptr<i32>, %cmp : i32, %new : i32) {
  // CHECK: llvm.cmpxchg %{{.*}}, %{{.*}}, %{{.*}} acq_rel monotonic : i32
  %0 = llvm.cmpxchg %ptr, %cmp, %new acq_rel monotonic : i32
  llvm.return
}

llvm.mlir.global external constant @_ZTIi() : !llvm.ptr<i8>
llvm.func @bar(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>)
llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: @invokeLandingpad
llvm.func @invokeLandingpad() -> i32 attributes { personality = @__gxx_personality_v0 } {
// CHECK: %[[a0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %{{.*}} = llvm.mlir.constant(3 : i32) : i32
// CHECK: %[[a2:.*]] = llvm.mlir.constant("\01") : !llvm.array<1 x i8>
// CHECK: %[[a3:.*]] = llvm.mlir.null : !llvm.ptr<ptr<i8>>
// CHECK: %[[a4:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK: %[[a5:.*]] = llvm.mlir.addressof @_ZTIi : !llvm.ptr<ptr<i8>>
// CHECK: %[[a6:.*]] = llvm.bitcast %[[a5]] : !llvm.ptr<ptr<i8>> to !llvm.ptr<i8>
// CHECK: %[[a7:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[a8:.*]] = llvm.alloca %[[a7]] x i8 : (i32) -> !llvm.ptr<i8>
// CHECK: %{{.*}} = llvm.invoke @foo(%[[a7]]) to ^[[BB2:.*]] unwind ^[[BB1:.*]] : (i32) -> !llvm.struct<(i32, f64, i32)>
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(3 : i32) : i32
  %2 = llvm.mlir.constant("\01") : !llvm.array<1 x i8>
  %3 = llvm.mlir.null : !llvm.ptr<ptr<i8>>
  %4 = llvm.mlir.null : !llvm.ptr<i8>
  %5 = llvm.mlir.addressof @_ZTIi : !llvm.ptr<ptr<i8>>
  %6 = llvm.bitcast %5 : !llvm.ptr<ptr<i8>> to !llvm.ptr<i8>
  %7 = llvm.mlir.constant(1 : i32) : i32
  %8 = llvm.alloca %7 x i8 : (i32) -> !llvm.ptr<i8>
  %9 = llvm.invoke @foo(%7) to ^bb2 unwind ^bb1 : (i32) -> !llvm.struct<(i32, f64, i32)>

// CHECK: ^[[BB1]]:
// CHECK:   %[[lp:.*]] = llvm.landingpad cleanup (catch %[[a3]] : !llvm.ptr<ptr<i8>>) (catch %[[a6]] : !llvm.ptr<i8>) (filter %[[a2]] : !llvm.array<1 x i8>) : !llvm.struct<(ptr<i8>, i32)>
// CHECK:   %{{.*}} = llvm.intr.eh.typeid.for %6 : i32
// CHECK:   llvm.resume %[[lp]] : !llvm.struct<(ptr<i8>, i32)>
^bb1:
  %10 = llvm.landingpad cleanup (catch %3 : !llvm.ptr<ptr<i8>>) (catch %6 : !llvm.ptr<i8>) (filter %2 : !llvm.array<1 x i8>) : !llvm.struct<(ptr<i8>, i32)>
  %11 = llvm.intr.eh.typeid.for %6 : i32
  llvm.resume %10 : !llvm.struct<(ptr<i8>, i32)>

// CHECK: ^[[BB2]]:
// CHECK:   llvm.return %[[a7]] : i32
^bb2:
  llvm.return %7 : i32

// CHECK: ^[[BB3:.*]]:
// CHECK:   llvm.invoke @bar(%[[a8]], %[[a6]], %[[a4]]) to ^[[BB2]] unwind ^[[BB1]] : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
^bb3:
  llvm.invoke @bar(%8, %6, %4) to ^bb2 unwind ^bb1 : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> ()

// CHECK: ^[[BB4:.*]]:
// CHECK:   llvm.return %[[a0]] : i32
^bb4:
  llvm.return %0 : i32
}

// CHECK-LABEL: @useFreezeOp
func.func @useFreezeOp(%arg0: i32) {
  // CHECK:  = llvm.freeze %[[ARG0:.*]] : i32
  %0 = llvm.freeze %arg0 : i32
  // CHECK: %[[x:.*]] = llvm.mlir.undef : i8
  %1 = llvm.mlir.undef : i8
  // CHECK:  = llvm.freeze %[[x]] : i8
  %2 = llvm.freeze %1 : i8
  return
}

// CHECK-LABEL: @useFenceInst
func.func @useFenceInst() {
  // CHECK:  syncscope("agent") seq_cst
  llvm.fence syncscope("agent") seq_cst
  // CHECK:  seq_cst
  llvm.fence syncscope("") seq_cst
  // CHECK:  release
  llvm.fence release
  return
}

// CHECK-LABEL: @useInlineAsm
llvm.func @useInlineAsm(%arg0: i32) {
  //      CHECK:  llvm.inline_asm {{.*}} (i32) -> i8
  %0 = llvm.inline_asm "bswap $0", "=r,r" %arg0 : (i32) -> i8

  // CHECK-NEXT:  llvm.inline_asm {{.*}} (i32, i32) -> i8
  %1 = llvm.inline_asm "foo", "bar" %arg0, %arg0 : (i32, i32) -> i8

  // CHECK-NEXT:  llvm.inline_asm has_side_effects {{.*}} (i32, i32) -> i8
  %2 = llvm.inline_asm has_side_effects "foo", "bar" %arg0, %arg0 : (i32, i32) -> i8

  // CHECK-NEXT:  llvm.inline_asm is_align_stack {{.*}} (i32, i32) -> i8
  %3 = llvm.inline_asm is_align_stack "foo", "bar" %arg0, %arg0 : (i32, i32) -> i8

  // CHECK-NEXT:  llvm.inline_asm "foo", "=r,=r,r" {{.*}} : (i32) -> !llvm.struct<(i8, i8)>
  %5 = llvm.inline_asm "foo", "=r,=r,r" %arg0 : (i32) -> !llvm.struct<(i8, i8)>

  llvm.return
}

// CHECK-LABEL: @fastmathFlags
func.func @fastmathFlags(%arg0: f32, %arg1: f32, %arg2: i32) {
// CHECK: {{.*}} = llvm.fadd %arg0, %arg1 {fastmathFlags = #llvm.fastmath<fast>} : f32
// CHECK: {{.*}} = llvm.fsub %arg0, %arg1 {fastmathFlags = #llvm.fastmath<fast>} : f32
// CHECK: {{.*}} = llvm.fmul %arg0, %arg1 {fastmathFlags = #llvm.fastmath<fast>} : f32
// CHECK: {{.*}} = llvm.fdiv %arg0, %arg1 {fastmathFlags = #llvm.fastmath<fast>} : f32
// CHECK: {{.*}} = llvm.frem %arg0, %arg1 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %0 = llvm.fadd %arg0, %arg1 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %1 = llvm.fsub %arg0, %arg1 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %2 = llvm.fmul %arg0, %arg1 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %3 = llvm.fdiv %arg0, %arg1 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %4 = llvm.frem %arg0, %arg1 {fastmathFlags = #llvm.fastmath<fast>} : f32

// CHECK: {{.*}} = llvm.fcmp "oeq" %arg0, %arg1 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %5 = llvm.fcmp "oeq" %arg0, %arg1 {fastmathFlags = #llvm.fastmath<fast>} : f32

// CHECK: {{.*}} = llvm.fneg %arg0 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %6 = llvm.fneg %arg0 {fastmathFlags = #llvm.fastmath<fast>} : f32

// CHECK: {{.*}} = llvm.call @foo(%arg2) {fastmathFlags = #llvm.fastmath<fast>} : (i32) -> !llvm.struct<(i32, f64, i32)>
  %7 = llvm.call @foo(%arg2) {fastmathFlags = #llvm.fastmath<fast>} : (i32) -> !llvm.struct<(i32, f64, i32)>

// CHECK: {{.*}} = llvm.fadd %arg0, %arg1 : f32
  %8 = llvm.fadd %arg0, %arg1 {fastmathFlags = #llvm.fastmath<>} : f32
// CHECK: {{.*}} = llvm.fadd %arg0, %arg1 {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f32
  %9 = llvm.fadd %arg0, %arg1 {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f32

// CHECK: {{.*}} = llvm.fneg %arg0 : f32
  %10 = llvm.fneg %arg0 {fastmathFlags = #llvm.fastmath<>} : f32
  return
}

module {
  // CHECK-LABEL: @loopOptions
  llvm.func @loopOptions() {
    // CHECK: llvm.br
    // CHECK-SAME: llvm.loop = {options = #llvm.loopopts<disable_unroll = true, disable_licm = true, interleave_count = 1, disable_pipeline = true, pipeline_initiation_interval = 1>}, parallel_access = [@metadata::@group1]}
    llvm.br ^bb1 {llvm.loop = {options = #llvm.loopopts<disable_unroll = true, disable_licm = true, interleave_count = 1, disable_pipeline = true, pipeline_initiation_interval = 1>}, parallel_access = [@metadata::@group1]}
  ^bb1:
    llvm.return
  }
  // CHECK: llvm.metadata @metadata attributes {test_attribute} {
  llvm.metadata @metadata attributes {test_attribute} {
    // CHECK: llvm.access_group @group1
    llvm.access_group @group1
    llvm.return
  }
}

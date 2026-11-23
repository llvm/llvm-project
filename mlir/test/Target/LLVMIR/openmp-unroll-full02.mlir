// Test lowering of omp.unroll_full applied to the inner loop of a nest
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @unroll_full_inner_loop(%baseptr: !llvm.ptr) -> () {
  %tc1 = llvm.mlir.constant(4 : i32) : i32
  %tc2 = llvm.mlir.constant(8 : i32) : i32
  %outer_cli = omp.new_cli
  %inner_cli = omp.new_cli
  omp.canonical_loop(%outer_cli) %iv1 : i32 in range(%tc1) {
    omp.canonical_loop(%inner_cli) %iv2 : i32 in range(%tc2) {
      %ptr = llvm.getelementptr inbounds %baseptr[%iv2] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %val = llvm.mlir.constant(42.0 : f32) : f32
      llvm.store %val, %ptr : f32, !llvm.ptr
      omp.terminator
    }
    omp.terminator
  }
  omp.unroll_full(%inner_cli)
  llvm.return
}

// CHECK-LABEL: define void @unroll_full_inner_loop(
// CHECK-SAME:      ptr %[[PTR:.*]])

// The outer loop counts to 4 ...
// CHECK:       %[[OUTER_IV:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[OUTER_NEXT:.*]], %[[OUTER_INC:.*]] ]
// CHECK:       icmp ult i32 %[[OUTER_IV]], 4

// ... and the inner loop, which holds the store, counts to 8.
// CHECK:       %[[INNER_IV:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[INNER_NEXT:.*]], %[[INNER_INC:.*]] ]
// CHECK:       icmp ult i32 %[[INNER_IV]], 8
// CHECK:       %[[GEP:.*]] = getelementptr inbounds float, ptr %[[PTR]], i32 %[[INNER_IV]]
// CHECK:       store float 4.200000e+01, ptr %[[GEP]]

// The full-unroll metadata hangs off the inner loop's backedge.
// CHECK:     [[INNER_INC]]:
// CHECK-NEXT:  %[[INNER_NEXT]] = add nuw i32 %[[INNER_IV]], 1
// CHECK-NEXT:  br label %{{.*}}, !llvm.loop ![[MD:[0-9]+]]

// The outer backedge carries none; the trailing anchor is what keeps an
// accidental !llvm.loop here from passing.
// CHECK:     [[OUTER_INC]]:
// CHECK-NEXT:  %[[OUTER_NEXT]] = add nuw i32 %[[OUTER_IV]], 1
// CHECK-NEXT:  br label %{{.*}}{{$}}

// CHECK:       ![[MD]] = distinct !{![[MD]], ![[ENABLE:[0-9]+]], ![[FULL:[0-9]+]]}
// CHECK-DAG:   ![[ENABLE]] = !{!"llvm.loop.unroll.enable"}
// CHECK-DAG:   ![[FULL]] = !{!"llvm.loop.unroll.full"}

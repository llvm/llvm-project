// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Regression test for inscan reduction combiner operand ordering. The combiner
// here is intentionally NONcommutative (subtraction) so that reversing the two
// operands passed to the reduction combiner changes the emitted IR. This pins
// the operand order at the two combiner call sites of the log-scan lowering:
// the orig-val seed and the prefix computation. A commutative combiner such as
// `+` produces identical IR regardless of operand order and would not catch a
// reversal, so this complements openmp-reduction-scan.mlir. The scan combines
// elements left-to-right, so the earlier prefix element must be the
// accumulator (`omp_out`, the combiner's first operand) and the later element
// the incoming value (`omp_in`, the second operand).

omp.declare_reduction @sub_reduction_i32 : i32 init {
^bb0(%arg0: i32):
  %0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%0 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  %0 = llvm.sub %arg0, %arg1 : i32
  omp.yield(%0 : i32)
}
// CHECK-LABEL: @scan_reduction_noncommutative
llvm.func @scan_reduction_noncommutative() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "z"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "y"} : (i64) -> !llvm.ptr
  %5 = llvm.alloca %0 x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr
  %7 = llvm.alloca %0 x i32 {bindc_name = "k"} : (i64) -> !llvm.ptr
  %10 = llvm.mlir.constant(100 : i32) : i32
  %11 = llvm.mlir.constant(1 : i32) : i32
  %12 = llvm.mlir.constant(0 : i32) : i32
  %13 = llvm.mlir.constant(100 : index) : i64
  %14 = llvm.mlir.addressof @_QFEa : !llvm.ptr
  %15 = llvm.mlir.addressof @_QFEb : !llvm.ptr
  omp.parallel {
    %37 = llvm.mlir.constant(1 : i64) : i64
    %38 = llvm.alloca %37 x i32 {bindc_name = "k", pinned} : (i64) -> !llvm.ptr
    %39 = llvm.mlir.constant(1 : i64) : i64
    omp.wsloop reduction(mod: inscan, @sub_reduction_i32 %5 -> %arg0 : !llvm.ptr) {
      omp.loop_nest (%arg1) : i32 = (%11) to (%10) inclusive step (%11) {
        llvm.store %arg1, %38 : i32, !llvm.ptr
        %40 = llvm.load %arg0 : !llvm.ptr -> i32
        %41 = llvm.load %38 : !llvm.ptr -> i32
        %42 = llvm.sext %41 : i32 to i64
        %50 = llvm.getelementptr %14[%42] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %51 = llvm.load %50 : !llvm.ptr -> i32
        %52 = llvm.add %40, %51 : i32
        llvm.store %52, %arg0 : i32, !llvm.ptr
        omp.scan inclusive(%arg0 : !llvm.ptr)
        llvm.store %arg1, %38 : i32, !llvm.ptr
        %53 = llvm.load %arg0 : !llvm.ptr -> i32
        %54 = llvm.load %38 : !llvm.ptr -> i32
        %55 = llvm.sext %54 : i32 to i64
        %63 = llvm.getelementptr %15[%55] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %53, %63 : i32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}
llvm.mlir.global internal @_QFEa() {addr_space = 0 : i32} : !llvm.array<100 x i32> {
  %0 = llvm.mlir.zero : !llvm.array<100 x i32>
  llvm.return %0 : !llvm.array<100 x i32>
}
llvm.mlir.global internal @_QFEb() {addr_space = 0 : i32} : !llvm.array<100 x i32> {
  %0 = llvm.mlir.zero : !llvm.array<100 x i32>
  llvm.return %0 : !llvm.array<100 x i32>
}

// The orig-val seed must combine as sub(orig-val, buffer[1]): the original
// variable's incoming value (the leftmost prefix element) is the accumulator
// (first operand) and buffer[1] is the incoming value (second operand).
//CHECK: omp.scan.seed:
//CHECK:   %[[OBUFF:.+]] = load ptr, ptr %{{.*}}, align 8
//CHECK:   %[[OELEMPTR:.+]] = getelementptr inbounds i32, ptr %[[OBUFF]], i32 1
//CHECK:   %[[OORIG:.+]] = load i32, ptr %{{.*}}, align 4
//CHECK:   %[[OELEM:.+]] = load i32, ptr %[[OELEMPTR]], align 4
//CHECK:   %[[OCOMB:.+]] = sub i32 %[[OORIG]], %[[OELEM]]
//CHECK:   store i32 %[[OCOMB]], ptr %[[OELEMPTR]], align 4

// The prefix computation must combine as sub(buffer[i-pow2k], buffer[i]): the
// earlier partial prefix is the accumulator (first operand) and the later one
// the incoming value (second operand). buffer[i] (the later slot) is updated.
//CHECK: omp.inner.log.scan.body:
//CHECK:   %[[IND1:.+]] = add i32 %{{.*}}, 1
//CHECK:   %[[IND1PTR:.+]] = getelementptr inbounds i32, ptr %{{.*}}, i32 %[[IND1]]
//CHECK:   %[[IND2:.+]] = sub nuw i32 %[[IND1]], %{{.*}}
//CHECK:   %[[IND2PTR:.+]] = getelementptr inbounds i32, ptr %{{.*}}, i32 %[[IND2]]
//CHECK:   %[[IND1VAL:.+]] = load i32, ptr %[[IND1PTR]], align 4
//CHECK:   %[[IND2VAL:.+]] = load i32, ptr %[[IND2PTR]], align 4
//CHECK:   %[[REDVAL:.+]] = sub i32 %[[IND2VAL]], %[[IND1VAL]]
//CHECK:   store i32 %[[REDVAL]], ptr %[[IND1PTR]], align 4

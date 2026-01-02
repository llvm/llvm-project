// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s
omp.declare_reduction @add_reduction_i32 : i32 init {
^bb0(%arg0: i32):
  %0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%0 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  %0 = llvm.add %arg0, %arg1 : i32
  omp.yield(%0 : i32)
}
// CHECK-LABEL: @scan_reduction
llvm.func @scan_reduction() {
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
    omp.wsloop reduction(mod: inscan, @add_reduction_i32 %5 -> %arg0 : !llvm.ptr) {
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
//CHECK: %vla = alloca ptr, align 8
//CHECK: omp_parallel
//CHECK: store ptr %vla, ptr %gep_vla, align 8
//CHECK: @__kmpc_fork_call
//CHECK: void @scan_reduction..omp_par
//CHECK: %[[BUFF_PTR:.+]] = load ptr, ptr %gep_vla
//CHECK: @__kmpc_masked
//CHECK: @__kmpc_barrier
//CHECK: @__kmpc_masked
//CHECK: @__kmpc_barrier
//CHECK: omp.scan.loop.cont:
//CHECK: @__kmpc_masked
//CHECK: @__kmpc_barrier
//CHECK: %[[FREE_VAR:.+]] = load ptr, ptr %[[BUFF_PTR]], align 8
//CHECK:  %[[ARRLAST:.+]] = getelementptr inbounds i32, ptr %[[FREE_VAR]], i32 100
//CHECK:  %[[RES:.+]] = load i32, ptr %[[ARRLAST]], align 4
//CHECK:  store i32 %[[RES]], ptr %loadgep{{.*}}, align 4
//CHECK: tail call void @free(ptr %[[FREE_VAR]])
//CHECK: @__kmpc_end_masked
//CHECK: omp.inscan.dispatch{{.*}}:                            ; preds = %omp_loop.body{{.*}}
//CHECK:   %[[BUFFVAR:.+]] = load ptr, ptr %[[BUFF_PTR]], align 8
//CHECK:   %[[arrayOffset1:.+]] = getelementptr inbounds i32, ptr %[[BUFFVAR]], i32 %{{.*}}
//CHECK:   %[[BUFFVAL1:.+]] = load i32, ptr %[[arrayOffset1]], align 4
//CHECK:   store i32 %[[BUFFVAL1]], ptr %{{.*}}, align 4
//CHECK:   %[[LOG:.+]] = call double @llvm.log2.f64(double 1.000000e+02) #0
//CHECK:   %[[CEIL:.+]] = call double @llvm.ceil.f64(double %[[LOG]]) #0
//CHECK:   %[[UB:.+]] = fptoui double %[[CEIL]] to i32
//CHECK:   br label %omp.outer.log.scan.body
//CHECK: omp.outer.log.scan.body:
//CHECK:   %[[K:.+]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXTK:.+]], %omp.inner.log.scan.exit ]
//CHECK:   %[[I:.+]] = phi i32 [ 1, %{{.*}} ], [ %[[NEXTI:.+]], %omp.inner.log.scan.exit ]
//CHECK:   %[[CMP1:.+]] = icmp uge i32 99, %[[I]]
//CHECK:   br i1 %[[CMP1]], label %omp.inner.log.scan.body, label %omp.inner.log.scan.exit
//CHECK: omp.inner.log.scan.exit:                          ; preds = %omp.inner.log.scan.body, %omp.outer.log.scan.body
//CHECK:   %[[NEXTK]] = add nuw i32 %[[K]], 1
//CHECK:   %[[NEXTI]] = shl nuw i32 %[[I]], 1
//CHECK:   %[[CMP2:.+]] = icmp ne i32 %[[NEXTK]], %[[UB]]
//CHECK:   br i1 %[[CMP2]], label %omp.outer.log.scan.body, label %omp.outer.log.scan.exit
//CHECK: omp.outer.log.scan.exit:                          ; preds = %omp.inner.log.scan.exit
//CHECK: @__kmpc_end_masked
//CHECK: omp.inner.log.scan.body:                          ; preds = %omp.inner.log.scan.body, %omp.outer.log.scan.body
//CHECK:   %[[CNT:.+]] = phi i32 [ 99, %omp.outer.log.scan.body ], [ %[[CNTNXT:.+]], %omp.inner.log.scan.body ]
//CHECK:   %[[BUFF:.+]] = load ptr, ptr %[[BUFF_PTR]]
//CHECK:   %[[IND1:.+]] = add i32 %[[CNT]], 1
//CHECK:   %[[IND1PTR:.+]] = getelementptr inbounds i32, ptr %[[BUFF]], i32 %[[IND1]]
//CHECK:   %[[IND2:.+]] = sub nuw i32 %[[IND1]], %[[I]]
//CHECK:   %[[IND2PTR:.+]] = getelementptr inbounds i32, ptr %[[BUFF]], i32 %[[IND2]]
//CHECK:   %[[IND1VAL:.+]] = load i32, ptr %[[IND1PTR]], align 4
//CHECK:   %[[IND2VAL:.+]] = load i32, ptr %[[IND2PTR]], align 4
//CHECK:   %[[REDVAL:.+]] = add i32 %[[IND1VAL]], %[[IND2VAL]]
//CHECK:   store i32 %[[REDVAL]], ptr %[[IND1PTR]], align 4
//CHECK:   %[[CNTNXT]] = sub nuw i32 %[[CNT]], 1
//CHECK:   %[[CMP3:.+]] = icmp uge i32 %[[CNTNXT]], %[[I]]
//CHECK:   br i1 %[[CMP3]], label %omp.inner.log.scan.body, label %omp.inner.log.scan.exit
//CHECK: omp.inscan.dispatch:                              ; preds = %omp_loop.body
//CHECK:   br i1 true, label %omp.before.scan.bb, label %omp.after.scan.bb
//CHECK: omp.loop_nest.region:                             ; preds = %omp.before.scan.bb
//CHECK:   %[[BUFFER:.+]] = load ptr, ptr %loadgep_vla, align 8
//CHECK:   %[[ARRAYOFFSET2:.+]] = getelementptr inbounds i32, ptr %[[BUFFER]], i32 %{{.*}}
//CHECK-NEXT:   %[[REDPRIVVAL:.+]] = load i32, ptr %{{.*}}, align 4
//CHECK:   store i32 %[[REDPRIVVAL]], ptr %[[ARRAYOFFSET2]], align 4
//CHECK:   br label %omp.scan.loop.exit

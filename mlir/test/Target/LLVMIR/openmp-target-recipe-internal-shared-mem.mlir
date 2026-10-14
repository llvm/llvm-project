// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Verifies that stack allocations emitted inside omp.private init regions and
// omp.declare_reduction alloc/init regions, when inlined in a Generic target
// device context and observed across threads, are placed in device shared
// memory (__kmpc_alloc_shared + __kmpc_free_shared) instead.

module attributes {omp.is_target_device = true, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<5> = dense<32> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, "dlti.alloca_memory_space" = 5 : ui64, "dlti.global_memory_space" = 1 : ui64>} {

  // The alloc region allocates an internal buffer, the init region allocates a
  // scratch descriptor copy and zero-initializes the neutral value in a loop,
  // and the combiner allocates scratch copies of both operands and sums them
  // element-wise in a loop.
  omp.declare_reduction @red.reducer byref_element_type(!llvm.array<4 x i32>) : !llvm.ptr alloc {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %desc = llvm.alloca %c1 x !llvm.array<4 x i32> : (i64) -> !llvm.ptr<5>
    %desccast = llvm.addrspacecast %desc : !llvm.ptr<5> to !llvm.ptr
    omp.yield(%desccast : !llvm.ptr)
  } init {
  ^bb0(%mold: !llvm.ptr, %alloc: !llvm.ptr):
    %zero = llvm.mlir.constant(0 : i32) : i32
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c4 = llvm.mlir.constant(4 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %tmp = llvm.alloca %c1 x !llvm.array<4 x i32> : (i64) -> !llvm.ptr<5>
    %tmpcast = llvm.addrspacecast %tmp : !llvm.ptr<5> to !llvm.ptr
    llvm.br ^bb1(%c0 : i64)
  ^bb1(%iv: i64):
    %cond = llvm.icmp "slt" %iv, %c4 : i64
    llvm.cond_br %cond, ^bb2, ^bb3
  ^bb2:
    %p = llvm.getelementptr %tmpcast[0, %iv] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
    llvm.store %zero, %p : i32, !llvm.ptr
    %next = llvm.add %iv, %c1 : i64
    llvm.br ^bb1(%next : i64)
  ^bb3:
    "llvm.intr.memcpy"(%alloc, %tmpcast, %c16) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    omp.yield(%alloc : !llvm.ptr)
  } combiner {
  ^bb0(%a: !llvm.ptr, %b: !llvm.ptr):
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c4 = llvm.mlir.constant(4 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %sa = llvm.alloca %c1 x !llvm.array<4 x i32> : (i64) -> !llvm.ptr<5>
    %sacast = llvm.addrspacecast %sa : !llvm.ptr<5> to !llvm.ptr
    %sb = llvm.alloca %c1 x !llvm.array<4 x i32> : (i64) -> !llvm.ptr<5>
    %sbcast = llvm.addrspacecast %sb : !llvm.ptr<5> to !llvm.ptr
    "llvm.intr.memcpy"(%sacast, %a, %c16) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    "llvm.intr.memcpy"(%sbcast, %b, %c16) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.br ^bb1(%c0 : i64)
  ^bb1(%iv: i64):
    %cond = llvm.icmp "slt" %iv, %c4 : i64
    llvm.cond_br %cond, ^bb2, ^bb3
  ^bb2:
    %pa = llvm.getelementptr %sacast[0, %iv] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
    %pb = llvm.getelementptr %sbcast[0, %iv] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
    %va = llvm.load %pa : !llvm.ptr -> i32
    %vb = llvm.load %pb : !llvm.ptr -> i32
    %sum = llvm.add %va, %vb : i32
    %pdst = llvm.getelementptr %a[0, %iv] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
    llvm.store %sum, %pdst : i32, !llvm.ptr
    %next = llvm.add %iv, %c1 : i64
    llvm.br ^bb1(%next : i64)
  ^bb3:
    omp.yield(%a : !llvm.ptr)
  }

  // The alloc region allocates a descriptor and the init region allocates
  // runtime-sized backing storage that is yielded out.
  omp.declare_reduction @arr_red.reducer byref_element_type(i32) : !llvm.ptr alloc {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %desc = llvm.alloca %c1 x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr<5>
    %desccast = llvm.addrspacecast %desc : !llvm.ptr<5> to !llvm.ptr
    omp.yield(%desccast : !llvm.ptr)
  } init {
  ^bb0(%mold: !llvm.ptr, %alloc: !llvm.ptr):
    %n = llvm.load %mold : !llvm.ptr -> i64
    %data = llvm.alloca %n x i32 : (i64) -> !llvm.ptr<5>
    %datacast = llvm.addrspacecast %data : !llvm.ptr<5> to !llvm.ptr
    llvm.store %datacast, %alloc : !llvm.ptr, !llvm.ptr
    omp.yield(%alloc : !llvm.ptr)
  } combiner {
  ^bb0(%a: !llvm.ptr, %b: !llvm.ptr):
    omp.yield(%a : !llvm.ptr)
  }

  // A privatizer whose init region allocates internal storage and stores its
  // address into the private box.
  omp.private {type = private} @box.privatizer : !llvm.ptr init {
  ^bb0(%mold: !llvm.ptr, %priv: !llvm.ptr):
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %data = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr<5>
    %datacast = llvm.addrspacecast %data : !llvm.ptr<5> to !llvm.ptr
    llvm.store %datacast, %priv : !llvm.ptr, !llvm.ptr
    omp.yield(%priv : !llvm.ptr)
  }

  // A privatizer whose init region optionally allocates runtime-sized internal
  // storage.
  omp.private {type = private} @rt_box.privatizer : !llvm.ptr init {
  ^bb0(%mold: !llvm.ptr, %priv: !llvm.ptr):
    %n = llvm.load %mold : !llvm.ptr -> i64
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %cond = llvm.icmp "eq" %n, %c0 : i64
    llvm.cond_br %cond, ^empty, ^alloc
  ^empty:
    llvm.br ^done
  ^alloc:
    %data = llvm.alloca %n x i32 : (i64) -> !llvm.ptr<5>
    %datacast = llvm.addrspacecast %data : !llvm.ptr<5> to !llvm.ptr
    llvm.store %datacast, %priv : !llvm.ptr, !llvm.ptr
    llvm.br ^done
  ^done:
    omp.yield(%priv : !llvm.ptr)
  }

  // CHECK-LABEL: define {{.*}} void @__omp_offloading_{{.*}}_private_init_l
  // CHECK: %[[PTR_SLOT:omp\.shared\.ptr.*]] = alloca ptr, {{.*}}addrspace(5)
  // CHECK: %[[SIZE_SLOT:omp\.shared\.size.*]] = alloca i64, {{.*}}addrspace(5)
  // CHECK: store ptr null, ptr addrspace(5) %[[PTR_SLOT]]
  // CHECK-NEXT: store i64 0, ptr addrspace(5) %[[SIZE_SLOT]]
  // CHECK: omp.private.init{{.*}}:
  // CHECK-NEXT: %[[DATA:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 4)
  // CHECK-NEXT: store ptr %[[DATA]], ptr addrspace(5) %[[PTR_SLOT]]
  // CHECK-NEXT: store i64 4, ptr addrspace(5) %[[SIZE_SLOT]]
  // CHECK-NOT: alloca i32
  // CHECK: omp.par.exit{{.*}}:
  // CHECK: br label %[[TEST_BB:.*]]
  // CHECK: [[TEST_BB]]:
  // CHECK-NEXT: %[[LOAD_PTR:.*]] = load ptr, ptr addrspace(5) %[[PTR_SLOT]]
  // CHECK-NEXT: %[[CMP:.*]] = icmp ne ptr %[[LOAD_PTR]], null
  // CHECK-NEXT: br i1 %[[CMP]], label %[[FREE_BB:.*]], label %[[CONT_BB:.*]]
  // CHECK: [[FREE_BB]]:
  // CHECK-NEXT: %[[LOAD_SIZE:.*]] = load i64, ptr addrspace(5) %[[SIZE_SLOT]]
  // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[LOAD_PTR]], i64 %[[LOAD_SIZE]])
  // CHECK-NEXT: br label %[[CONT_BB]]
  llvm.func @private_init() attributes {omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    omp.target kernel_type(generic) private(@box.privatizer %2 -> %arg0 : !llvm.ptr) {
      omp.parallel {
        %3 = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }

  // CHECK-LABEL: define {{.*}} void @__omp_offloading_{{.*}}_teams_reduction_l
  // CHECK: %[[I_PTR:omp\.shared\.ptr[0-9]*]] = alloca ptr, {{.*}}addrspace(5)
  // CHECK: %[[I_SIZE:omp\.shared\.size[0-9]*]] = alloca i64, {{.*}}addrspace(5)
  // CHECK: %[[A_PTR:omp\.shared\.ptr[0-9]*]] = alloca ptr, {{.*}}addrspace(5)
  // CHECK: %[[A_SIZE:omp\.shared\.size[0-9]*]] = alloca i64, {{.*}}addrspace(5)
  // CHECK: store ptr null, ptr addrspace(5) %[[I_PTR]]
  // CHECK-NEXT: store i64 0, ptr addrspace(5) %[[I_SIZE]]
  // CHECK: store ptr null, ptr addrspace(5) %[[A_PTR]]
  // CHECK-NEXT: store i64 0, ptr addrspace(5) %[[A_SIZE]]
  // CHECK: %[[A_MEM:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 16)
  // CHECK-NEXT: store ptr %[[A_MEM]], ptr addrspace(5) %[[A_PTR]]
  // CHECK-NEXT: store i64 16, ptr addrspace(5) %[[A_SIZE]]
  // CHECK: omp.reduction.init{{.*}}:
  // CHECK: br label %[[RED_NEUTRAL_BB:.*]]
  // CHECK: [[RED_NEUTRAL_BB]]:
  // CHECK-NEXT: %[[I_MEM:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 16)
  // CHECK-NEXT: store ptr %[[I_MEM]], ptr addrspace(5) %[[I_PTR]]
  // CHECK-NEXT: store i64 16, ptr addrspace(5) %[[I_SIZE]]
  // CHECK: reduce.finalize{{.*}}:
  // CHECK-NEXT: %[[LI:.*]] = load ptr, ptr addrspace(5) %[[I_PTR]]
  // CHECK-NEXT: %[[CI:.*]] = icmp ne ptr %[[LI]], null
  // CHECK-NEXT: br i1 %[[CI]], label %[[FREE_BB1:.*]], label %[[CONT_BB1:.*]]
  // CHECK: [[FREE_BB1]]:
  // CHECK-NEXT: %[[LIS:.*]] = load i64, ptr addrspace(5) %[[I_SIZE]]
  // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[LI]], i64 %[[LIS]])
  // CHECK-NEXT: br label %[[CONT_BB1]]
  // CHECK: [[CONT_BB1]]:
  // CHECK-NEXT: %[[LA:.*]] = load ptr, ptr addrspace(5) %[[A_PTR]]
  // CHECK-NEXT: %[[CA:.*]] = icmp ne ptr %[[LA]], null
  // CHECK-NEXT: br i1 %[[CA]], label %[[FREE_BB2:.*]], label %[[CONT_BB2:.*]]
  // CHECK: [[FREE_BB2]]:
  // CHECK-NEXT: %[[LAS:.*]] = load i64, ptr addrspace(5) %[[A_SIZE]]
  // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[LA]], i64 %[[LAS]])
  // CHECK-NEXT: br label %[[CONT_BB2]]
  llvm.func @teams_reduction() attributes {omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>} {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %0 = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr<5>
    %1 = llvm.addrspacecast %0 : !llvm.ptr<5> to !llvm.ptr
    %m = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%m -> %arg0 : !llvm.ptr) {
      omp.teams reduction(byref @red.reducer %arg0 -> %arg1 : !llvm.ptr) {
        omp.parallel {
          %l = llvm.load %arg1 : !llvm.ptr -> i32
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }

  // CHECK-LABEL: define {{.*}} void @__omp_offloading_{{.*}}_private_init_runtime_l
  // CHECK: %[[PTR_SLOT:omp\.shared\.ptr[0-9]*]] = alloca ptr, {{.*}}addrspace(5)
  // CHECK: %[[SIZE_SLOT:omp\.shared\.size[0-9]*]] = alloca i64, {{.*}}addrspace(5)
  // CHECK: store ptr null, ptr addrspace(5) %[[PTR_SLOT]]
  // CHECK-NEXT: store i64 0, ptr addrspace(5) %[[SIZE_SLOT]]
  // CHECK: %[[SZ:.*]] = mul i64 4, %{{.*}}
  // CHECK-NEXT: %[[DATA:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 %[[SZ]])
  // CHECK-NEXT: store ptr %[[DATA]], ptr addrspace(5) %[[PTR_SLOT]]
  // CHECK-NEXT: store i64 %[[SZ]], ptr addrspace(5) %[[SIZE_SLOT]]
  // CHECK: omp.par.exit{{.*}}:
  // CHECK: br label %[[GUARD_BB:.*]]
  // CHECK: [[GUARD_BB]]:
  // CHECK-NEXT: %[[P:.*]] = load ptr, ptr addrspace(5) %[[PTR_SLOT]]
  // CHECK-NEXT: %[[NN:.*]] = icmp ne ptr %[[P]], null
  // CHECK-NEXT: br i1 %[[NN]], label %[[FREE_BB:.*]], label %[[CONT_BB:.*]]
  // CHECK: [[FREE_BB]]:
  // CHECK-NEXT: %[[LSZ:.*]] = load i64, ptr addrspace(5) %[[SIZE_SLOT]]
  // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[P]], i64 %[[LSZ]])
  // CHECK-NEXT: br label %[[CONT_BB]]
  llvm.func @private_init_runtime() attributes {omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %m = omp.map.info var_ptr(%2 : !llvm.ptr, !llvm.ptr) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%m -> %arg0 : !llvm.ptr) private(@rt_box.privatizer %2 -> %arg1 [map_idx=0] : !llvm.ptr) {
      omp.parallel {
        %3 = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }

  // CHECK-LABEL: define {{.*}} void @__omp_offloading_{{.*}}_teams_reduction_array_l
  // CHECK: %[[I_PTR:omp\.shared\.ptr[0-9]*]] = alloca ptr, {{.*}}addrspace(5)
  // CHECK: %[[I_SIZE:omp\.shared\.size[0-9]*]] = alloca i64, {{.*}}addrspace(5)
  // CHECK: %[[A_PTR:omp\.shared\.ptr[0-9]*]] = alloca ptr, {{.*}}addrspace(5)
  // CHECK: %[[A_SIZE:omp\.shared\.size[0-9]*]] = alloca i64, {{.*}}addrspace(5)
  // CHECK: store ptr null, ptr addrspace(5) %[[I_PTR]]
  // CHECK-NEXT: store i64 0, ptr addrspace(5) %[[I_SIZE]]
  // CHECK: store ptr null, ptr addrspace(5) %[[A_PTR]]
  // CHECK-NEXT: store i64 0, ptr addrspace(5) %[[A_SIZE]]
  // CHECK: %[[A_MEM:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 16)
  // CHECK-NEXT: store ptr %[[A_MEM]], ptr addrspace(5) %[[A_PTR]]
  // CHECK-NEXT: store i64 16, ptr addrspace(5) %[[A_SIZE]]
  // CHECK: omp.reduction.init{{.*}}:
  // CHECK: %[[RN:.*]] = load i64, ptr {{.*}}
  // CHECK-NEXT: %[[RSZ:.*]] = mul i64 4, %[[RN]]
  // CHECK-NEXT: %[[I_MEM:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 %[[RSZ]])
  // CHECK-NEXT: store ptr %[[I_MEM]], ptr addrspace(5) %[[I_PTR]]
  // CHECK-NEXT: store i64 %[[RSZ]], ptr addrspace(5) %[[I_SIZE]]
  // CHECK: reduce.finalize{{.*}}:
  // CHECK-NEXT: %[[LI:.*]] = load ptr, ptr addrspace(5) %[[I_PTR]]
  // CHECK-NEXT: %[[CI:.*]] = icmp ne ptr %[[LI]], null
  // CHECK-NEXT: br i1 %[[CI]], label %[[FREE_BB1:.*]], label %[[CONT_BB1:.*]]
  // CHECK: [[FREE_BB1]]:
  // CHECK-NEXT: %[[LIS:.*]] = load i64, ptr addrspace(5) %[[I_SIZE]]
  // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[LI]], i64 %[[LIS]])
  // CHECK-NEXT: br label %[[CONT_BB1]]
  // CHECK: [[CONT_BB1]]:
  // CHECK-NEXT: %[[LA:.*]] = load ptr, ptr addrspace(5) %[[A_PTR]]
  // CHECK-NEXT: %[[CA:.*]] = icmp ne ptr %[[LA]], null
  // CHECK-NEXT: br i1 %[[CA]], label %[[FREE_BB2:.*]], label %[[CONT_BB2:.*]]
  // CHECK: [[FREE_BB2]]:
  // CHECK-NEXT: %[[LAS:.*]] = load i64, ptr addrspace(5) %[[A_SIZE]]
  // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[LA]], i64 %[[LAS]])
  // CHECK-NEXT: br label %[[CONT_BB2]]
  llvm.func @teams_reduction_array() attributes {omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>} {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %0 = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr<5>
    %1 = llvm.addrspacecast %0 : !llvm.ptr<5> to !llvm.ptr
    %m = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%m -> %arg0 : !llvm.ptr) {
      omp.teams reduction(byref @arr_red.reducer %arg0 -> %arg1 : !llvm.ptr) {
        omp.parallel {
          %l = llvm.load %arg1 : !llvm.ptr -> i32
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }

  // CHECK-LABEL: define {{.*}} void @__omp_offloading_{{.*}}_distribute_reduction_l{{.*}}omp_par
  // CHECK: teams.alloca:
  // CHECK: %[[I_PTR:omp\.shared\.ptr[0-9]*]] = alloca ptr, {{.*}}addrspace(5)
  // CHECK: %[[I_SIZE:omp\.shared\.size[0-9]*]] = alloca i64, {{.*}}addrspace(5)
  // CHECK: store ptr null, ptr addrspace(5) %[[I_PTR]]
  // CHECK-NEXT: store i64 0, ptr addrspace(5) %[[I_SIZE]]
  // CHECK: %[[A_PTR:omp\.shared\.ptr[0-9]*]] = alloca ptr, {{.*}}addrspace(5)
  // CHECK: %[[A_SIZE:omp\.shared\.size[0-9]*]] = alloca i64, {{.*}}addrspace(5)
  // CHECK: store ptr null, ptr addrspace(5) %[[A_PTR]]
  // CHECK-NEXT: store i64 0, ptr addrspace(5) %[[A_SIZE]]
  // CHECK: %[[A_MEM:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 16)
  // CHECK-NEXT: store ptr %[[A_MEM]], ptr addrspace(5) %[[A_PTR]]
  // CHECK-NEXT: store i64 16, ptr addrspace(5) %[[A_SIZE]]
  // CHECK: omp.reduction.init{{.*}}:
  // CHECK: br label %[[RED_NEUTRAL_BB:.*]]
  // CHECK: [[RED_NEUTRAL_BB]]:
  // CHECK-NEXT: %[[I_MEM:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 16)
  // CHECK-NEXT: store ptr %[[I_MEM]], ptr addrspace(5) %[[I_PTR]]
  // CHECK-NEXT: store i64 16, ptr addrspace(5) %[[I_SIZE]]
  // CHECK: reduce.finalize{{.*}}:
  // CHECK-NEXT: %[[LI:.*]] = load ptr, ptr addrspace(5) %[[I_PTR]]
  // CHECK-NEXT: %[[CI:.*]] = icmp ne ptr %[[LI]], null
  // CHECK-NEXT: br i1 %[[CI]], label %[[FREE_I:.*]], label %[[GUARD_A:.*]]
  // CHECK: [[GUARD_A]]:
  // CHECK-NEXT: %[[LA:.*]] = load ptr, ptr addrspace(5) %[[A_PTR]]
  // CHECK-NEXT: %[[CA:.*]] = icmp ne ptr %[[LA]], null
  // CHECK-NEXT: br i1 %[[CA]], label %[[FREE_A:.*]], label %[[CONT_A:.*]]
  // CHECK: [[FREE_A]]:
  // CHECK-NEXT: %[[LAS:.*]] = load i64, ptr addrspace(5) %[[A_SIZE]]
  // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[LA]], i64 %[[LAS]])
  // CHECK-NEXT: br label %[[CONT_A]]
  // CHECK: [[FREE_I]]:
  // CHECK-NEXT: %[[LIS:.*]] = load i64, ptr addrspace(5) %[[I_SIZE]]
  // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[LI]], i64 %[[LIS]])
  // CHECK-NEXT: br label %[[GUARD_A]]
  // CHECK: omp.reduction.nonatomic.body{{.*}}:
  // CHECK-NOT: br
  // CHECK: alloca [4 x i32]
  // CHECK-NOT: br
  // CHECK: alloca [4 x i32]
  llvm.func @extfunc(!llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>}
  llvm.func @distribute_reduction() attributes {omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>} {
    %cc1 = llvm.mlir.constant(1 : i32) : i32
    %0 = llvm.alloca %cc1 x i32 : (i32) -> !llvm.ptr<5>
    %1 = llvm.addrspacecast %0 : !llvm.ptr<5> to !llvm.ptr
    %m = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%m -> %arg0 : !llvm.ptr) {
      %c1 = llvm.mlir.constant(1 : i32) : i32
      %c10 = llvm.mlir.constant(10 : i32) : i32
      omp.teams reduction(byref @red.reducer %arg0 -> %arg1 : !llvm.ptr) {
        omp.distribute {
          omp.loop_nest (%iv) : i32 = (%c1) to (%c10) inclusive step (%c1) {
            llvm.call @extfunc(%arg1) : (!llvm.ptr) -> ()
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }

  // CHECK-LABEL: define {{.*}} void @__omp_offloading_{{.*}}_distribute_private_l{{.*}}omp_par
  // CHECK: distribute.alloca:
  // CHECK-NOT: br
  // CHECK: %[[RT_PTR:omp\.shared\.ptr[0-9]*]] = alloca ptr, {{.*}}addrspace(5)
  // CHECK: %[[RT_SIZE:omp\.shared\.size[0-9]*]] = alloca i64, {{.*}}addrspace(5)
  // CHECK: store ptr null, ptr addrspace(5) %[[RT_PTR]]
  // CHECK-NEXT: store i64 0, ptr addrspace(5) %[[RT_SIZE]]
  // CHECK: %[[FX_PTR:omp\.shared\.ptr[0-9]*]] = alloca ptr, {{.*}}addrspace(5)
  // CHECK: %[[FX_SIZE:omp\.shared\.size[0-9]*]] = alloca i64, {{.*}}addrspace(5)
  // CHECK: store ptr null, ptr addrspace(5) %[[FX_PTR]]
  // CHECK-NEXT: store i64 0, ptr addrspace(5) %[[FX_SIZE]]
  // CHECK: omp.private.init{{.*}}:
  // CHECK-NEXT: %[[FX_MEM:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 4)
  // CHECK-NEXT: store ptr %[[FX_MEM]], ptr addrspace(5) %[[FX_PTR]]
  // CHECK-NEXT: store i64 4, ptr addrspace(5) %[[FX_SIZE]]
  // CHECK-NOT: br
  // CHECK: br label %[[INIT_GUARD_BB:.*]]
  // CHECK: [[INIT_GUARD_BB]]:
  // CHECK-NEXT: %[[RN:.*]] = load i64
  // CHECK-NEXT: %[[ISZ:.*]] = icmp eq i64 %[[RN]], 0
  // CHECK-NEXT: br i1 %[[ISZ]], label %{{.*}}, label %[[RT_ALLOC_BB:.*]]
  // CHECK: [[RT_ALLOC_BB]]:
  // CHECK-NEXT: %[[SZ:.*]] = mul i64 4, %[[RN]]
  // CHECK-NEXT: %[[RT_MEM:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 %[[SZ]])
  // CHECK-NEXT: store ptr %[[RT_MEM]], ptr addrspace(5) %[[RT_PTR]]
  // CHECK-NEXT: store i64 %[[SZ]], ptr addrspace(5) %[[RT_SIZE]]
  // CHECK: omp_loop.after{{.*}}:
  // CHECK-NEXT: br label %[[GUARD_FX:.*]]
  // CHECK: [[GUARD_FX]]:
  // CHECK-NEXT: %[[LFX:.*]] = load ptr, ptr addrspace(5) %[[FX_PTR]]
  // CHECK-NEXT: %[[CFX:.*]] = icmp ne ptr %[[LFX]], null
  // CHECK-NEXT: br i1 %[[CFX]], label %[[FREE_FX:.*]], label %[[GUARD_RT:.*]]
  // CHECK: [[GUARD_RT]]:
  // CHECK-NOT: br
  // CHECK: %[[LRT:.*]] = load ptr, ptr addrspace(5) %[[RT_PTR]]
  // CHECK-NEXT: %[[CRT:.*]] = icmp ne ptr %[[LRT]], null
  // CHECK-NEXT: br i1 %[[CRT]], label %[[FREE_RT:.*]], label %[[CONT_RT:.*]]
  // CHECK: [[FREE_RT]]:
  // CHECK-NEXT: %[[LRTS:.*]] = load i64, ptr addrspace(5) %[[RT_SIZE]]
  // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[LRT]], i64 %[[LRTS]])
  // CHECK-NEXT: br label %[[CONT_RT]]
  // CHECK: [[FREE_FX]]:
  // CHECK-NEXT: %[[LFXS:.*]] = load i64, ptr addrspace(5) %[[FX_SIZE]]
  // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[LFX]], i64 %[[LFXS]])
  // CHECK-NEXT: br label %[[GUARD_RT]]
  llvm.func @distribute_private() attributes {omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %3 = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr<5>
    %4 = llvm.addrspacecast %3 : !llvm.ptr<5> to !llvm.ptr
    %m1 = omp.map.info var_ptr(%2 : !llvm.ptr, !llvm.ptr) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %m2 = omp.map.info var_ptr(%4 : !llvm.ptr, !llvm.ptr) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%m1 -> %arg0, %m2 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      %c1 = llvm.mlir.constant(1 : i32) : i32
      %c10 = llvm.mlir.constant(10 : i32) : i32
      omp.teams {
        omp.distribute private(@box.privatizer %arg0 -> %arg2, @rt_box.privatizer %arg1 -> %arg3 : !llvm.ptr, !llvm.ptr) {
          omp.loop_nest (%iv) : i32 = (%c1) to (%c10) inclusive step (%c1) {
            llvm.call @extfunc(%arg2) : (!llvm.ptr) -> ()
            llvm.call @extfunc(%arg3) : (!llvm.ptr) -> ()
            omp.yield
          }
        }
        omp.terminator
      } {omp.combined}
      omp.terminator
    }
    llvm.return
  }

  // CHECK-LABEL: define {{.*}} void @__omp_offloading_{{.*}}_private_init_noshare_l
  // CHECK-NOT: omp.shared.ptr
  // CHECK-NOT: @__kmpc_alloc_shared
  // CHECK: ret void
  llvm.func @private_init_noshare() attributes {omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    omp.target kernel_type(generic) private(@box.privatizer %2 -> %arg0 : !llvm.ptr) {
      %3 = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Prefix compare+capture: {read, compare}
// CHECK-LABEL: define void @compare_capture_prefix(
// CHECK-SAME:    ptr %[[X:.*]], ptr %[[E:.*]], ptr %[[D:.*]], ptr %[[V:.*]])
// CHECK:         %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
// CHECK:         %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
// CHECK:         %[[RESULT:.*]] = cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] monotonic monotonic
// CHECK:         %[[OLD:.*]] = extractvalue { i32, i1 } %[[RESULT]], 0
// CHECK:         store i32 %[[OLD]], ptr %[[V]], align 4
llvm.func @compare_capture_prefix(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> i32
  %dval = llvm.load %d : !llvm.ptr -> i32
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, i32
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %cmp = llvm.icmp "eq" %xval, %eval : i32
      %sel = llvm.select %cmp, %dval, %xval : i1, i32
      omp.yield(%sel : i32)
    }
  }
  llvm.return
}

// Postfix compare+capture: {compare, read}
// CHECK-LABEL: define void @compare_capture_postfix(
// CHECK-SAME:    ptr %[[X:.*]], ptr %[[E:.*]], ptr %[[D:.*]], ptr %[[V:.*]])
// CHECK:         %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
// CHECK:         %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
// CHECK:         %[[RESULT:.*]] = cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] monotonic monotonic
// CHECK:         %[[OLD:.*]] = extractvalue { i32, i1 } %[[RESULT]], 0
// CHECK:         %[[SUCCESS:.*]] = extractvalue { i32, i1 } %[[RESULT]], 1
// CHECK:         %[[NEWVAL:.*]] = select i1 %[[SUCCESS]], i32 %[[DVAL]], i32 %[[OLD]]
// CHECK:         store i32 %[[NEWVAL]], ptr %[[V]], align 4
llvm.func @compare_capture_postfix(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> i32
  %dval = llvm.load %d : !llvm.ptr -> i32
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %cmp = llvm.icmp "eq" %xval, %eval : i32
      %sel = llvm.select %cmp, %dval, %xval : i1, i32
      omp.yield(%sel : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, i32
  }
  llvm.return
}

// Fail-only compare+capture: {compare, read} with fail_only
// CHECK-LABEL: define void @compare_capture_failonly(
// CHECK-SAME:    ptr %[[X:.*]], ptr %[[E:.*]], ptr %[[D:.*]], ptr %[[V:.*]])
// CHECK:         %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
// CHECK:         %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
// CHECK:         %[[RESULT:.*]] = cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] monotonic monotonic
// CHECK:         %[[OLD:.*]] = extractvalue { i32, i1 } %[[RESULT]], 0
// CHECK:         %[[SUCCESS:.*]] = extractvalue { i32, i1 } %[[RESULT]], 1
// CHECK:         br i1 %[[SUCCESS]], label %{{.*}}.atomic.exit, label %{{.*}}.atomic.cont
// CHECK:       {{.*}}.atomic.cont:
// CHECK:         store i32 %[[OLD]], ptr %[[V]]
// CHECK:         br label %{{.*}}.atomic.exit
// CHECK:       {{.*}}.atomic.exit:
llvm.func @compare_capture_failonly(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> i32
  %dval = llvm.load %d : !llvm.ptr -> i32
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %cmp = llvm.icmp "eq" %xval, %eval : i32
      %sel = llvm.select %cmp, %dval, %xval : i1, i32
      omp.yield(%sel : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, i32
  } fail_only
  llvm.return
}

// Fail-only compare+capture with an explicit `fail` clause: V is captured on
// the failure path, and the fail clause sets that path's ordering. Success
// order seq_cst, fail order acquire => `cmpxchg ... seq_cst acquire`.
// CHECK-LABEL: define void @compare_capture_failonly_fail(
// CHECK-SAME:    ptr %[[X:.*]], ptr %[[E:.*]], ptr %[[D:.*]], ptr %[[V:.*]])
// CHECK:         %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
// CHECK:         %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
// CHECK:         %[[RESULT:.*]] = cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] seq_cst acquire
// CHECK:         %[[OLD:.*]] = extractvalue { i32, i1 } %[[RESULT]], 0
// CHECK:         %[[SUCCESS:.*]] = extractvalue { i32, i1 } %[[RESULT]], 1
// CHECK:         br i1 %[[SUCCESS]], label %{{.*}}.atomic.exit, label %{{.*}}.atomic.cont
// CHECK:       {{.*}}.atomic.cont:
// CHECK:         store i32 %[[OLD]], ptr %[[V]]
// CHECK:         br label %{{.*}}.atomic.exit
// CHECK:       {{.*}}.atomic.exit:
llvm.func @compare_capture_failonly_fail(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> i32
  %dval = llvm.load %d : !llvm.ptr -> i32
  omp.atomic.capture memory_order(seq_cst) {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %cmp = llvm.icmp "eq" %xval, %eval : i32
      %sel = llvm.select %cmp, %dval, %xval : i1, i32
      omp.yield(%sel : i32)
    } fail_memory_order(acquire)
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, i32
  } fail_only
  llvm.return
}

// Weak compare+capture: {read, compare} with weak
// CHECK-LABEL: define void @compare_capture_weak(
// CHECK-SAME:    ptr %[[X:.*]], ptr %[[E:.*]], ptr %[[D:.*]], ptr %[[V:.*]])
// CHECK:         %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
// CHECK:         %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
// CHECK:         %[[RESULT:.*]] = cmpxchg weak ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] monotonic monotonic
// CHECK:         %[[OLD:.*]] = extractvalue { i32, i1 } %[[RESULT]], 0
// CHECK:         store i32 %[[OLD]], ptr %[[V]], align 4
llvm.func @compare_capture_weak(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> i32
  %dval = llvm.load %d : !llvm.ptr -> i32
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, i32
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %cmp = llvm.icmp "eq" %xval, %eval : i32
      %sel = llvm.select %cmp, %dval, %xval : i1, i32
      omp.yield(%sel : i32)
    } weak
  }
  llvm.return
}

// ===== Float (real) tests =====
// Float uses the HandleFPNegZero path with NaN/zero checks and bitcast.

// Float prefix compare+capture: v gets old value
// CHECK-LABEL: define void @compare_capture_float_prefix(
// CHECK:         cmpxchg ptr
// CHECK:       {{.*}}.atomic.exit:
// CHECK:         store float %{{.*}}, ptr %[[V:.*]], align 4
llvm.func @compare_capture_float_prefix(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> f32
  %dval = llvm.load %d : !llvm.ptr -> f32
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, f32
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: f32):
      %cmp = llvm.fcmp "oeq" %xval, %eval : f32
      %sel = llvm.select %cmp, %dval, %xval : i1, f32
      omp.yield(%sel : f32)
    }
  }
  llvm.return
}

// Float postfix compare+capture: v gets select(success, d, old)
// CHECK-LABEL: define void @compare_capture_float_postfix(
// CHECK:         cmpxchg ptr
// CHECK:       {{.*}}.atomic.exit:
// CHECK:         %[[OK:.*]] = phi i1
// CHECK:         %[[OLD_FP:.*]] = bitcast i32 %{{.*}} to float
// CHECK:         %[[NEW:.*]] = select i1 %[[OK]], float %{{.*}}, float %[[OLD_FP]]
// CHECK:         store float %[[NEW]], ptr %{{.*}}, align 4
llvm.func @compare_capture_float_postfix(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> f32
  %dval = llvm.load %d : !llvm.ptr -> f32
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: f32):
      %cmp = llvm.fcmp "oeq" %xval, %eval : f32
      %sel = llvm.select %cmp, %dval, %xval : i1, f32
      omp.yield(%sel : f32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, f32
  }
  llvm.return
}

// Float fail-only compare+capture: conditional store on failure
// CHECK-LABEL: define void @compare_capture_float_failonly(
// CHECK:         cmpxchg ptr
// CHECK:       {{.*}}.atomic.exit:
// CHECK:         %[[OK:.*]] = phi i1
// CHECK:         %[[OLD_FP:.*]] = bitcast i32 %{{.*}} to float
// CHECK:         br i1 %[[OK]], label %{{.*}}.atomic.exit{{.*}}, label %{{.*}}.atomic.cont
// CHECK:       {{.*}}.atomic.cont:
// CHECK:         store float %[[OLD_FP]], ptr %{{.*}}
llvm.func @compare_capture_float_failonly(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> f32
  %dval = llvm.load %d : !llvm.ptr -> f32
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: f32):
      %cmp = llvm.fcmp "oeq" %xval, %eval : f32
      %sel = llvm.select %cmp, %dval, %xval : i1, f32
      omp.yield(%sel : f32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, f32
  } fail_only
  llvm.return
}

// ===== Complex (struct<(f32, f32)>) tests =====
// Complex uses bitcast to i64 for cmpxchg, with FP neg-zero handling.

// Complex prefix compare+capture: v gets old value
// CHECK-LABEL: define void @compare_capture_complex_prefix(
// CHECK:         cmpxchg ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}} monotonic monotonic
// CHECK:         store { float, float } %{{.*}}, ptr %{{.*}}
llvm.func @compare_capture_complex_prefix(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> !llvm.struct<(f32, f32)>
  %dval = llvm.load %d : !llvm.ptr -> !llvm.struct<(f32, f32)>
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, !llvm.struct<(f32, f32)>
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: !llvm.struct<(f32, f32)>):
      %xr = llvm.extractvalue %xval[0] : !llvm.struct<(f32, f32)>
      %er = llvm.extractvalue %eval[0] : !llvm.struct<(f32, f32)>
      %cmpr = llvm.fcmp "oeq" %xr, %er : f32
      %xi = llvm.extractvalue %xval[1] : !llvm.struct<(f32, f32)>
      %ei = llvm.extractvalue %eval[1] : !llvm.struct<(f32, f32)>
      %cmpi = llvm.fcmp "oeq" %xi, %ei : f32
      %cmp = llvm.and %cmpr, %cmpi : i1
      %sel = llvm.select %cmp, %dval, %xval : i1, !llvm.struct<(f32, f32)>
      omp.yield(%sel : !llvm.struct<(f32, f32)>)
    }
  }
  llvm.return
}

// Complex postfix compare+capture: v gets select(success, d, old)
// CHECK-LABEL: define void @compare_capture_complex_postfix(
// CHECK:         cmpxchg ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}} monotonic monotonic
// CHECK:         select i1 %{{.*}}, { float, float } %{{.*}}, { float, float } %{{.*}}
// CHECK:         store { float, float } %{{.*}}, ptr %{{.*}}
llvm.func @compare_capture_complex_postfix(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> !llvm.struct<(f32, f32)>
  %dval = llvm.load %d : !llvm.ptr -> !llvm.struct<(f32, f32)>
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: !llvm.struct<(f32, f32)>):
      %xr = llvm.extractvalue %xval[0] : !llvm.struct<(f32, f32)>
      %er = llvm.extractvalue %eval[0] : !llvm.struct<(f32, f32)>
      %cmpr = llvm.fcmp "oeq" %xr, %er : f32
      %xi = llvm.extractvalue %xval[1] : !llvm.struct<(f32, f32)>
      %ei = llvm.extractvalue %eval[1] : !llvm.struct<(f32, f32)>
      %cmpi = llvm.fcmp "oeq" %xi, %ei : f32
      %cmp = llvm.and %cmpr, %cmpi : i1
      %sel = llvm.select %cmp, %dval, %xval : i1, !llvm.struct<(f32, f32)>
      omp.yield(%sel : !llvm.struct<(f32, f32)>)
    }
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, !llvm.struct<(f32, f32)>
  }
  llvm.return
}

// Complex fail-only compare+capture: conditional store on failure
// CHECK-LABEL: define void @compare_capture_complex_failonly(
// CHECK:         cmpxchg ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}} monotonic monotonic
// CHECK:         br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK:         store { float, float } %{{.*}}, ptr %{{.*}}
llvm.func @compare_capture_complex_failonly(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> !llvm.struct<(f32, f32)>
  %dval = llvm.load %d : !llvm.ptr -> !llvm.struct<(f32, f32)>
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: !llvm.struct<(f32, f32)>):
      %xr = llvm.extractvalue %xval[0] : !llvm.struct<(f32, f32)>
      %er = llvm.extractvalue %eval[0] : !llvm.struct<(f32, f32)>
      %cmpr = llvm.fcmp "oeq" %xr, %er : f32
      %xi = llvm.extractvalue %xval[1] : !llvm.struct<(f32, f32)>
      %ei = llvm.extractvalue %eval[1] : !llvm.struct<(f32, f32)>
      %cmpi = llvm.fcmp "oeq" %xi, %ei : f32
      %cmp = llvm.and %cmpr, %cmpi : i1
      %sel = llvm.select %cmp, %dval, %xval : i1, !llvm.struct<(f32, f32)>
      omp.yield(%sel : !llvm.struct<(f32, f32)>)
    }
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, !llvm.struct<(f32, f32)>
  } fail_only
  llvm.return
}

// Complex acquire compare+capture: when the component comparison fails we
// branch around the cmpxchg, so the initial atomic load (the only memory
// operation on the failure path) must carry the acquire failure ordering.
// CHECK-LABEL: define void @compare_capture_complex_acquire(
// CHECK:         load atomic i64, ptr %{{.*}} acquire
// CHECK:         cmpxchg ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}} acquire acquire
// CHECK:         store { float, float } %{{.*}}, ptr %{{.*}}
llvm.func @compare_capture_complex_acquire(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> !llvm.struct<(f32, f32)>
  %dval = llvm.load %d : !llvm.ptr -> !llvm.struct<(f32, f32)>
  omp.atomic.capture memory_order(acquire) {
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, !llvm.struct<(f32, f32)>
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: !llvm.struct<(f32, f32)>):
      %xr = llvm.extractvalue %xval[0] : !llvm.struct<(f32, f32)>
      %er = llvm.extractvalue %eval[0] : !llvm.struct<(f32, f32)>
      %cmpr = llvm.fcmp "oeq" %xr, %er : f32
      %xi = llvm.extractvalue %xval[1] : !llvm.struct<(f32, f32)>
      %ei = llvm.extractvalue %eval[1] : !llvm.struct<(f32, f32)>
      %cmpi = llvm.fcmp "oeq" %xi, %ei : f32
      %cmp = llvm.and %cmpr, %cmpi : i1
      %sel = llvm.select %cmp, %dval, %xval : i1, !llvm.struct<(f32, f32)>
      omp.yield(%sel : !llvm.struct<(f32, f32)>)
    }
  }
  llvm.return
}

// Complex seq_cst compare+capture: the failure path load must carry the
// seq_cst failure ordering.
// CHECK-LABEL: define void @compare_capture_complex_seqcst(
// CHECK:         load atomic i64, ptr %{{.*}} seq_cst
// CHECK:         cmpxchg ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}} seq_cst seq_cst
// CHECK:         store { float, float } %{{.*}}, ptr %{{.*}}
llvm.func @compare_capture_complex_seqcst(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> !llvm.struct<(f32, f32)>
  %dval = llvm.load %d : !llvm.ptr -> !llvm.struct<(f32, f32)>
  omp.atomic.capture memory_order(seq_cst) {
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, !llvm.struct<(f32, f32)>
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: !llvm.struct<(f32, f32)>):
      %xr = llvm.extractvalue %xval[0] : !llvm.struct<(f32, f32)>
      %er = llvm.extractvalue %eval[0] : !llvm.struct<(f32, f32)>
      %cmpr = llvm.fcmp "oeq" %xr, %er : f32
      %xi = llvm.extractvalue %xval[1] : !llvm.struct<(f32, f32)>
      %ei = llvm.extractvalue %eval[1] : !llvm.struct<(f32, f32)>
      %cmpi = llvm.fcmp "oeq" %xi, %ei : f32
      %cmp = llvm.and %cmpr, %cmpi : i1
      %sel = llvm.select %cmp, %dval, %xval : i1, !llvm.struct<(f32, f32)>
      omp.yield(%sel : !llvm.struct<(f32, f32)>)
    }
  }
  llvm.return
}

// ===== Weak clause with float =====

// Float weak prefix: cmpxchg weak with FP neg-zero handling
// CHECK-LABEL: define void @compare_capture_float_weak(
// CHECK:         cmpxchg weak ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}} monotonic monotonic
// CHECK:         store float %{{.*}}, ptr %{{.*}}
llvm.func @compare_capture_float_weak(%x : !llvm.ptr, %e : !llvm.ptr, %d : !llvm.ptr, %v : !llvm.ptr) {
  %eval = llvm.load %e : !llvm.ptr -> f32
  %dval = llvm.load %d : !llvm.ptr -> f32
  omp.atomic.capture memory_order(relaxed) {
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, f32
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval: f32):
      %cmp = llvm.fcmp "oeq" %xval, %eval : f32
      %sel = llvm.select %cmp, %dval, %xval : i1, f32
      omp.yield(%sel : f32)
    } weak
  }
  llvm.return
}

// ===== Min/Max compare+capture =====

// Integer min, postfix: atomicrmw min, v gets min(old, e)
// CHECK-LABEL: define void @compare_capture_min_postfix(
// CHECK-SAME:    ptr %[[X:.*]], ptr %[[V:.*]], i32 %[[E:.*]])
// CHECK:         %[[OLD:.*]] = atomicrmw min ptr %[[X]], i32 %[[E]] monotonic
// CHECK:         %[[NEW:.*]] = call i32 @llvm.smin.i32(i32 %[[OLD]], i32 %[[E]])
// CHECK:         store i32 %[[NEW]], ptr %[[V]]
llvm.func @compare_capture_min_postfix(%x : !llvm.ptr, %v : !llvm.ptr, %e : i32) {
  omp.atomic.capture {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval : i32):
      %cmp = llvm.icmp "sgt" %xval, %e : i32
      %sel = llvm.select %cmp, %e, %xval : i1, i32
      omp.yield(%sel : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, i32
  }
  llvm.return
}

// Integer max, postfix: atomicrmw max, v gets max(old, e)
// CHECK-LABEL: define void @compare_capture_max_postfix(
// CHECK:         %[[OLD:.*]] = atomicrmw max ptr %{{.*}}, i32 %[[E:.*]] monotonic
// CHECK:         %[[NEW:.*]] = call i32 @llvm.smax.i32(i32 %[[OLD]], i32 %[[E]])
// CHECK:         store i32 %[[NEW]], ptr %{{.*}}
llvm.func @compare_capture_max_postfix(%x : !llvm.ptr, %v : !llvm.ptr, %e : i32) {
  omp.atomic.capture {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval : i32):
      %cmp = llvm.icmp "slt" %xval, %e : i32
      %sel = llvm.select %cmp, %e, %xval : i1, i32
      omp.yield(%sel : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, i32
  }
  llvm.return
}

// Integer min, prefix: v gets old value
// CHECK-LABEL: define void @compare_capture_min_prefix(
// CHECK:         %[[OLD:.*]] = atomicrmw min ptr %{{.*}}, i32 %{{.*}} monotonic
// CHECK:         store i32 %[[OLD]], ptr %{{.*}}
llvm.func @compare_capture_min_prefix(%x : !llvm.ptr, %v : !llvm.ptr, %e : i32) {
  omp.atomic.capture {
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, i32
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval : i32):
      %cmp = llvm.icmp "sgt" %xval, %e : i32
      %sel = llvm.select %cmp, %e, %xval : i1, i32
      omp.yield(%sel : i32)
    }
  }
  llvm.return
}

// Float min, postfix: atomicrmw fmin, v gets minnum(old, e)
// CHECK-LABEL: define void @compare_capture_fmin_postfix(
// CHECK:         %[[OLD:.*]] = atomicrmw fmin ptr %{{.*}}, float %[[E:.*]] monotonic
// CHECK:         %[[NEW:.*]] = call float @llvm.minnum.f32(float %[[OLD]], float %[[E]])
// CHECK:         store float %[[NEW]], ptr %{{.*}}
llvm.func @compare_capture_fmin_postfix(%x : !llvm.ptr, %v : !llvm.ptr, %e : f32) {
  omp.atomic.capture {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval : f32):
      %cmp = llvm.fcmp "ogt" %xval, %e : f32
      %sel = llvm.select %cmp, %e, %xval : i1, f32
      omp.yield(%sel : f32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, f32
  }
  llvm.return
}

// Float max, postfix: atomicrmw fmax, v gets maxnum(old, e)
// CHECK-LABEL: define void @compare_capture_fmax_postfix(
// CHECK:         %[[OLD:.*]] = atomicrmw fmax ptr %{{.*}}, float %[[E:.*]] monotonic
// CHECK:         %[[NEW:.*]] = call float @llvm.maxnum.f32(float %[[OLD]], float %[[E]])
// CHECK:         store float %[[NEW]], ptr %{{.*}}
llvm.func @compare_capture_fmax_postfix(%x : !llvm.ptr, %v : !llvm.ptr, %e : f32) {
  omp.atomic.capture {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval : f32):
      %cmp = llvm.fcmp "olt" %xval, %e : f32
      %sel = llvm.select %cmp, %e, %xval : i1, f32
      omp.yield(%sel : f32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, f32
  }
  llvm.return
}

// ===== Min/Max fail-only =====

// Integer min, fail-only: conditional store on failure
// CHECK-LABEL: define void @compare_capture_min_fail_only(
// CHECK:         %[[OLD:.*]] = atomicrmw min ptr %{{.*}}, i32 %[[E:.*]] monotonic
// CHECK:         %[[UPD:.*]] = icmp sgt i32 %[[OLD]], %[[E]]
// CHECK:         %[[FAILED:.*]] = xor i1 %[[UPD]], true
// CHECK:         br i1 %[[FAILED]], label %[[CONT:.*]], label %[[EXIT:.*]]
// CHECK:       [[CONT]]:
// CHECK:         store i32 %[[OLD]], ptr %{{.*}}
llvm.func @compare_capture_min_fail_only(%x : !llvm.ptr, %v : !llvm.ptr, %e : i32) {
  omp.atomic.capture {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval : i32):
      %cmp = llvm.icmp "sgt" %xval, %e : i32
      %sel = llvm.select %cmp, %e, %xval : i1, i32
      omp.yield(%sel : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, i32
  } fail_only
  llvm.return
}

// Integer max, fail-only
// CHECK-LABEL: define void @compare_capture_max_fail_only(
// CHECK:         %[[OLD:.*]] = atomicrmw max ptr %{{.*}}, i32 %[[E:.*]] monotonic
// CHECK:         %[[UPD:.*]] = icmp slt i32 %[[OLD]], %[[E]]
// CHECK:         %[[FAILED:.*]] = xor i1 %[[UPD]], true
// CHECK:         br i1 %[[FAILED]], label %[[CONT:.*]], label %[[EXIT:.*]]
// CHECK:       [[CONT]]:
// CHECK:         store i32 %[[OLD]], ptr %{{.*}}
llvm.func @compare_capture_max_fail_only(%x : !llvm.ptr, %v : !llvm.ptr, %e : i32) {
  omp.atomic.capture {
    omp.atomic.compare %x : !llvm.ptr {
    ^bb0(%xval : i32):
      %cmp = llvm.icmp "slt" %xval, %e : i32
      %sel = llvm.select %cmp, %e, %xval : i1, i32
      omp.yield(%sel : i32)
    }
    omp.atomic.read %v = %x : !llvm.ptr, !llvm.ptr, i32
  } fail_only
  llvm.return
}

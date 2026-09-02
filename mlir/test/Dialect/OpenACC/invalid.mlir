// RUN: mlir-opt -split-input-file -verify-diagnostics %s

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
// expected-error@+1 {{gang, worker or vector cannot appear with seq}}
acc.loop gang control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} seq

// -----

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
// expected-error@+1 {{gang, worker or vector cannot appear with seq}}
acc.loop worker control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} seq

// -----

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
// expected-error@+1 {{gang, worker or vector cannot appear with seq}}
acc.loop vector control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} seq

// -----

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
// expected-error@+1 {{gang, worker or vector cannot appear with seq}}
acc.loop gang worker control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} seq

// -----

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
// expected-error@+1 {{gang, worker or vector cannot appear with seq}}
acc.loop gang vector control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} seq

// -----

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
// expected-error@+1 {{gang, worker or vector cannot appear with seq}}
acc.loop worker vector control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} seq

// -----

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
// expected-error@+1 {{gang, worker or vector cannot appear with seq}}
acc.loop gang worker vector {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} seq

// -----

// expected-error@+1 {{expected non-empty body.}}
acc.loop {
} independent

// -----

// expected-error@+1 {{'acc.loop' op duplicate device_type `none` found in gang attribute}}
acc.loop gang([#acc.device_type<none>, #acc.device_type<none>]) {
  acc.yield
}

// -----

// expected-error@+1 {{'acc.loop' op duplicate device_type `none` found in worker attribute}}
"acc.loop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, worker = [#acc.device_type<none>, #acc.device_type<none>]}> ({
  "acc.yield"() : () -> ()
}) : () -> ()

// -----

// expected-error@+1 {{'acc.loop' op duplicate device_type `none` found in vector attribute}}
"acc.loop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, vector = [#acc.device_type<none>, #acc.device_type<none>]}> ({
  "acc.yield"() : () -> ()
}) : () -> ()

// -----

// expected-error@+1 {{'acc.loop' op duplicate device_type `nvidia` found in gang attribute}}
acc.loop gang([#acc.device_type<nvidia>, #acc.device_type<nvidia>]) {
  acc.yield
}

// -----

// expected-error@+1 {{'acc.loop' op duplicate device_type `none` found in collapseDeviceType attribute}}
acc.loop {
  acc.yield
} collapse([1, 1]) collapseDeviceType([#acc.device_type<none>, #acc.device_type<none>]) independent

// -----

%i64value = arith.constant 1 : i64
// expected-error@+1 {{'acc.loop' op duplicate device_type `none` found in workerNumOperandsDeviceType attribute}}
acc.loop worker(%i64value: i64, %i64value: i64) {
  acc.yield
} independent

// -----

%i64value = arith.constant 1 : i64
// expected-error@+1 {{'acc.loop' op duplicate device_type `none` found in vectorOperandsDeviceType attribute}}
acc.loop vector(%i64value: i64, %i64value: i64) {
  acc.yield
} independent

// -----

func.func @acc_routine_bind() {
  return
}
// expected-error@+1 {{expected symbol reference or string attribute}}
acc.routine @acc_routine_bind_rout func(@acc_routine_bind) bind(42 : i64)

// -----

func.func @acc_routine_parallelism() -> () {
  return
}
// expected-error@+1 {{only one of `gang`, `worker`, `vector`, `seq` can be present at the same time for device_type `nvidia`}}
"acc.routine"() <{func_name = @acc_routine_parallelism, sym_name = "acc_routine_parallelism_rout", gang = [#acc.device_type<nvidia>], worker = [#acc.device_type<nvidia>]}> : () -> ()

// -----

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
// expected-error@+1 {{only one of auto, independent, seq can be present at the same time}}
acc.loop control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
  acc.yield
} auto_ seq inclusiveUpperbound(array<i1: true>)

// -----

// expected-error@+1 {{at least one operand or the default attribute must appear on the data operation}}
acc.data {
  acc.yield
}

// -----

%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{expect data entry/exit operation or acc.getdeviceptr as defining op}}
acc.data dataOperands(%value : memref<10xf32>) {
  acc.yield
}

// -----

// expected-error@+1 {{at least one value must be present in dataOperands}}
acc.update

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<f32>
%0 = acc.update_device varPtr(%value : memref<f32>) -> memref<f32>
// expected-error@+1 {{asyncOnly attribute cannot appear with asyncOperand}}
"acc.update"(%cst, %0) <{operandSegmentSizes = array<i32: 0, 1, 0, 1>, asyncOperandsDeviceType = [#acc.device_type<none>], asyncOnly = [#acc.device_type<none>]}> : (index, memref<f32>) -> ()

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<f32>
%0 = acc.update_device varPtr(%value : memref<f32>) -> memref<f32>
// expected-error@+1 {{wait attribute cannot appear with waitOperands}}
"acc.update"(%cst, %0) <{operandSegmentSizes = array<i32: 0, 0, 1, 1>, waitOperandsDeviceType = [#acc.device_type<none>], waitOperandsSegments = array<i32: 1>, hasWaitDevnum = [false], waitOnly = [#acc.device_type<none>]}> : (index, memref<f32>) -> ()

// -----

%cst = arith.constant 1 : index
// expected-error@+1 {{wait_devnum cannot appear without waitOperands}}
acc.wait wait_devnum(%cst: index)

// -----

%cst = arith.constant 1 : index
// expected-error@+1 {{async attribute cannot appear with asyncOperand}}
"acc.wait"(%cst) <{operandSegmentSizes = array<i32: 0, 1, 0, 0>, async}> : (index) -> ()

// -----

acc.parallel {
// expected-error@+1 {{'acc.init' op cannot be nested in a compute operation}}
  acc.init
  acc.yield
}

// -----

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
acc.loop control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32){
// expected-error@+1 {{'acc.init' op cannot be nested in a compute operation}}
  acc.init
  acc.yield
} inclusiveUpperbound(array<i1: true>) independent

// -----

acc.parallel {
// expected-error@+1 {{'acc.shutdown' op cannot be nested in a compute operation}}
  acc.shutdown
  acc.yield
}

// -----

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
acc.loop control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
// expected-error@+1 {{'acc.shutdown' op cannot be nested in a compute operation}}
  acc.shutdown
  acc.yield
} inclusiveUpperbound(array<i1: true>) independent

// -----

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
acc.loop control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
  "test.openacc_dummy_op"() ({
    // expected-error@+1 {{'acc.shutdown' op cannot be nested in a compute operation}}
    acc.shutdown
  }) : () -> ()
  acc.yield
} inclusiveUpperbound(array<i1: true>) independent

// -----

// expected-error@+1 {{at least one operand must be present in dataOperands on the exit data operation}}
acc.exit_data async

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<f32>
%0 = acc.getdeviceptr varPtr(%value : memref<f32>) -> memref<f32>
// expected-error@+1 {{async attribute cannot appear with asyncOperand}}
"acc.exit_data"(%cst, %0) <{operandSegmentSizes = array<i32: 0, 1, 0, 0, 1>, async}> : (index, memref<f32>) -> ()
acc.delete accPtr(%0 : memref<f32>)

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<f32>
%0 = acc.getdeviceptr varPtr(%value : memref<f32>) -> memref<f32>
// expected-error@+1 {{wait_devnum cannot appear without waitOperands}}
acc.exit_data wait_devnum(%cst: index) dataOperands(%0 : memref<f32>)
acc.delete accPtr(%0 : memref<f32>)

// -----

// expected-error@+1 {{at least one operand must be present in dataOperands on the enter data operation}}
acc.enter_data async

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<f32>
%0 = acc.create varPtr(%value : memref<f32>) -> memref<f32>
// expected-error@+1 {{async attribute cannot appear with asyncOperand}}
"acc.enter_data"(%cst, %0) <{operandSegmentSizes = array<i32: 0, 1, 0, 0, 1>, async}> : (index, memref<f32>) -> ()

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<f32>
%0 = acc.create varPtr(%value : memref<f32>) -> memref<f32>
// expected-error@+1 {{wait attribute cannot appear with waitOperands}}
"acc.enter_data"(%cst, %0) <{operandSegmentSizes = array<i32: 0, 0, 0, 1, 1>, wait}> : (index, memref<f32>) -> ()

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<f32>
%0 = acc.create varPtr(%value : memref<f32>) -> memref<f32>
// expected-error@+1 {{wait_devnum cannot appear without waitOperands}}
acc.enter_data wait_devnum(%cst: index) dataOperands(%0 : memref<f32>)

// -----

%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{expect data entry operation as defining op}}
acc.enter_data dataOperands(%value : memref<10xf32>)

// -----

%0 = arith.constant 1.0 : f32
// expected-error@+1 {{operand #0 must be integer or index, but got 'f32'}}
%1 = acc.bounds lowerbound(%0 : f32)

// -----

%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{expect data entry/exit operation or acc.getdeviceptr as defining op}}
acc.update dataOperands(%value : memref<10xf32>)

// -----

%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{expect data entry/exit operation or acc.getdeviceptr as defining op}}
acc.parallel dataOperands(%value : memref<10xf32>) {
  acc.yield
}

// -----

%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{expect data entry/exit operation or acc.getdeviceptr as defining op}}
acc.serial dataOperands(%value : memref<10xf32>) {
  acc.yield
}

// -----

%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{expect data entry/exit operation or acc.getdeviceptr as defining op}}
acc.kernels dataOperands(%value : memref<10xf32>) {
  acc.yield
}

// -----

// expected-error@+1 {{expects non-empty init region}}
acc.private.recipe @privatization_i32 : !llvm.ptr init {
}

// -----

// expected-error@+1 {{expects init region first argument of the privatization type}}
acc.private.recipe @privatization_i32 : !llvm.ptr init {
^bb0(%arg0 : i32):
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  acc.yield %0 : !llvm.ptr
}

// -----

// expected-error@+1 {{expects destroy region first argument of the privatization type}}
acc.private.recipe @privatization_i32 : !llvm.ptr init {
^bb0(%arg0 : !llvm.ptr):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %c0, %0 : i32, !llvm.ptr
  acc.yield %0 : !llvm.ptr
} destroy {
^bb0(%arg0 : f32):
  "test.openacc_dummy_op"(%arg0) : (f32) -> ()
}

// -----

// expected-error@+1 {{expects non-empty init region}}
acc.firstprivate.recipe @privatization_i32 : !llvm.ptr init {
} copy {}

// -----

// expected-error@+1 {{expects init region first argument of the privatization type}}
acc.firstprivate.recipe @privatization_i32 : !llvm.ptr init {
^bb0(%arg0 : i32):
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  acc.yield %0 : !llvm.ptr
} copy {}

// -----

// expected-error@+1 {{expects non-empty copy region}}
acc.firstprivate.recipe @privatization_i32 : !llvm.ptr init {
^bb0(%arg0 : !llvm.ptr):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %c0, %0 : i32, !llvm.ptr
  acc.yield %0 : !llvm.ptr
} copy {
}

// -----

// expected-error@+1 {{expects copy region with two arguments of the privatization type}}
acc.firstprivate.recipe @privatization_i32 : !llvm.ptr init {
^bb0(%arg0 : !llvm.ptr):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %c0, %0 : i32, !llvm.ptr
  acc.yield %0 : !llvm.ptr
} copy {
^bb0(%arg0 : f32):
  "test.openacc_dummy_op"(%arg0) : (f32) -> ()
}

// -----

// expected-error@+1 {{expects copy region with two arguments of the privatization type}}
acc.firstprivate.recipe @privatization_i32 : !llvm.ptr init {
^bb0(%arg0 : !llvm.ptr):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %c0, %0 : i32, !llvm.ptr
  acc.yield %0 : !llvm.ptr
} copy {
^bb0(%arg0 : f32, %arg1 : i32):
  "test.openacc_dummy_op"(%arg0) : (f32) -> ()
}

// -----

// expected-error@+1 {{expects destroy region first argument of the privatization type}}
acc.firstprivate.recipe @privatization_i32 : i32 init {
^bb0(%arg0 : i32):
  %0 = arith.constant 1 : i32
  acc.yield %0 : i32
} copy {
^bb0(%arg0 : i32, %arg1 : !llvm.ptr):
  llvm.store %arg0, %arg1 : i32, !llvm.ptr
  acc.yield
} destroy {
^bb0(%arg0 : f32):
  acc.yield
}

// -----

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
// expected-error@+1 {{expected ')'}}
acc.loop gang({static=%i64Value: i64, num=%i64Value: i64} control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
}

// -----

// expected-error@+1 {{expects non-empty init region}}
acc.reduction.recipe @reduction_i64 : i64 reduction_operator<add> init {
} combiner {}

// -----

// expected-error@+1 {{expects init region first argument of the reduction type}}
acc.reduction.recipe @reduction_i64 : i64 reduction_operator<add> init {
^bb0(%0: i32):
  %1 = arith.constant 0 : i64
  acc.yield %1 : i64
} combiner {}

// -----

// expected-error@+1 {{expects non-empty combiner region}}
acc.reduction.recipe @reduction_i64 : i64 reduction_operator<add> init {
^bb0(%0: i64):
  %1 = arith.constant 0 : i64
  acc.yield %1 : i64
} combiner {}

// -----

// expected-error@+1 {{expects combiner region with the first two arguments of the reduction type}}
acc.reduction.recipe @reduction_i64 : i64 reduction_operator<add> init {
^bb0(%0: i64):
  %1 = arith.constant 0 : i64
  acc.yield %1 : i64
} combiner {
^bb0(%0: i32):
  acc.yield %0 : i32
}

// -----

// expected-error@+1 {{expects combiner region with the first two arguments of the reduction type}}
acc.reduction.recipe @reduction_i64 : i64 reduction_operator<add> init {
^bb0(%0: i64):
  %1 = arith.constant 0 : i64
  acc.yield %1 : i64
} combiner {
^bb0(%0: i64):
  acc.yield %0 : i64
}

// -----

// expected-error@+1 {{expects combiner region to yield a value of the reduction type}}
acc.reduction.recipe @reduction_i64 : i64 reduction_operator<add> init {
^bb0(%0: i64):
  %1 = arith.constant 0 : i64
  acc.yield %1 : i64
} combiner {
^bb0(%0: i64, %1: i64):
  %2 = arith.constant 0 : i32
  acc.yield %2 : i32
}

// -----

%1 = arith.constant 1 : i32
%2 = arith.constant 10 : i32
// expected-error@+1 {{new value expected after comma}}
acc.loop gang({static=%i64Value: i64, ) control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
}

// -----

%i1 = arith.constant 1 : i32
%i2 = arith.constant 10 : i32
// expected-error@+1 {{unstructured acc.loop must not have induction variables}}
acc.loop control(%iv : i32) = (%i1 : i32) to (%i2 : i32) step (%i1 : i32) {
  acc.yield
} independent unstructured

// -----

// expected-error@+1 {{expect at least one of num, dim or static values}}
acc.loop gang({}) {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
}

// -----

%i64value = arith.constant 1 : i64
// expected-error@+1 {{num_gangs expects a maximum of 3 values per segment}}
acc.parallel num_gangs({%i64value: i64, %i64value : i64, %i64value : i64, %i64value : i64}) {
}

// -----

%0 = "arith.constant"() <{value = 1 : i64}> : () -> i64
// expected-error@+1 {{num_gangs operand count does not match count in segments}}
"acc.parallel"(%0) <{numGangsSegments = array<i32: 1>, operandSegmentSizes = array<i32: 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
}) : (i64) -> ()

// -----

%i64value = arith.constant 1 : i64
acc.parallel {
// expected-error@+1 {{'acc.set' op cannot be nested in a compute operation}}
  acc.set device_type(#acc.device_type<nvidia>)
  acc.yield
}

// -----

// expected-error@+1 {{'acc.set' op at least one default_async, device_num, or device_type operand must appear}}
acc.set

// -----

func.func @acc_atomic_write(%addr : memref<memref<i32>>, %val : i32) {
  // expected-error @below {{address must dereference to value type}}
  acc.atomic.write %addr = %val : memref<memref<i32>>, i32
  return
}

// -----

func.func @acc_atomic_update(%x: memref<i32>, %expr: f32) {
  // expected-error @below {{the type of the operand must be a pointer type whose element type is the same as that of the region argument}}
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: f32):
    %newval = llvm.fadd %xval, %expr : f32
    acc.yield %newval : f32
  }
  return
}

// -----

func.func @acc_atomic_update(%x: memref<i32>, %expr: i32) {
  // expected-error @+2 {{op expects regions to end with 'acc.yield', found 'acc.terminator'}}
  // expected-note @below {{in custom textual format, the absence of terminator implies 'acc.yield'}}
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    acc.terminator
  }
  return
}
// -----

func.func @acc_atomic_update(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{invalid kind of type specified}}
  acc.atomic.update %x : i32 {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    acc.yield %newval : i32
  }
  return
}

// -----

func.func @acc_atomic_update(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{only updated value must be returned}}
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    acc.yield %newval, %expr : i32, i32
  }
  return
}

// -----

func.func @acc_atomic_update(%x: memref<i32>, %expr: i32, %y: f32) {
  // expected-error @below {{input and yielded value must have the same type}}
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    acc.yield %y: f32
  }
  return
}

// -----

func.func @acc_atomic_update(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{the region must accept exactly one argument}}
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32, %tmp: i32):
    %newval = llvm.add %xval, %expr : i32
    acc.yield %newval : i32
  }
  return
}

// -----

func.func @acc_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  // expected-error @below {{expected three operations in atomic.capture region}}
  acc.atomic.capture {
    acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    acc.terminator
  }
  return
}

// -----

func.func @acc_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  acc.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    acc.terminator
  }
  return
}

// -----

func.func @acc_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  acc.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    acc.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      acc.yield %newval : i32
    }
    acc.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      acc.yield %newval : i32
    }
    acc.terminator
  }
  return
}

// -----

func.func @acc_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  acc.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    acc.atomic.write %x = %expr : memref<i32>, i32
    acc.atomic.write %x = %expr : memref<i32>, i32
    acc.terminator
  }
  return
}

// -----

func.func @acc_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  acc.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    acc.atomic.write %x = %expr : memref<i32>, i32
    acc.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      acc.yield %newval : i32
    }
    acc.terminator
  }
  return
}

// -----

func.func @acc_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  acc.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    acc.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      acc.yield %newval : i32
    }
    acc.atomic.write %x = %expr : memref<i32>, i32
    acc.terminator
  }
  return
}

// -----

func.func @acc_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  acc.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    acc.atomic.write %x = %expr : memref<i32>, i32
    acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    acc.terminator
  }
  return
}

// -----

func.func @acc_atomic_capture(%x: memref<i32>, %y: memref<i32>, %v: memref<i32>, %expr: i32) {
  acc.atomic.capture {
    // expected-error @below {{updated variable in atomic.update must be captured in second operation}}
    acc.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      acc.yield %newval : i32
    }
    acc.atomic.read %v = %y : memref<i32>, memref<i32>, i32
    acc.terminator
  }
}

// -----

func.func @acc_atomic_capture(%x: memref<i32>, %y: memref<i32>, %v: memref<i32>, %expr: i32) {
  acc.atomic.capture {
    // expected-error @below {{captured variable in atomic.read must be updated in second operation}}
    acc.atomic.read %v = %y : memref<i32>, memref<i32>, i32
    acc.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      acc.yield %newval : i32
    }
    acc.terminator
  }
}

// -----

func.func @acc_atomic_capture(%x: memref<i32>, %y: memref<i32>, %v: memref<i32>, %expr: i32) {
  acc.atomic.capture {
    // expected-error @below {{captured variable in atomic.read must be updated in second operation}}
    acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    acc.atomic.write %y = %expr : memref<i32>, i32
    acc.terminator
  }
}

// -----

func.func @acc_combined() {
  // expected-error @below {{expected 'loop'}}
  acc.parallel combined() {
  }
  return
}

// -----

func.func @acc_combined() {
  // expected-error @below {{expected compute construct name}}
  acc.loop combined(loop) {
  }
  return
}

// -----

func.func @acc_combined() {
  // expected-error @below {{expected 'loop'}}
  acc.parallel combined(parallel loop) {
  }
  return
}

// -----

func.func @acc_combined() {
  // expected-error @below {{expected ')'}}
  acc.loop combined(parallel loop) {
  }
  return
}

// -----

func.func @acc_loop_container() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  // expected-error @below {{found sibling loops inside container-like acc.loop}}
  acc.loop {
    scf.for %arg4 = %c0 to %c10 step %c1 {
      scf.yield
    }
    scf.for %arg5 = %c0 to %c10 step %c1 {
        scf.yield
    }
    acc.yield
  } collapse([2]) collapseDeviceType([#acc.device_type<none>]) independent
  return
}

// -----

func.func @acc_loop_container() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  // expected-error @below {{failed to find enough loop-like operations inside container-like acc.loop}}
  acc.loop {
    scf.for %arg4 = %c0 to %c10 step %c1 {
      scf.for %arg5 = %c0 to %c10 step %c1 {
          scf.yield
      }
      scf.yield
    }
    acc.yield
  } collapse([3]) collapseDeviceType([#acc.device_type<none>]) independent
  return
}

// -----

%value = memref.alloc() : memref<f32>
// expected-error @below {{no data clause modifiers are allowed}}
%0 = acc.private varPtr(%value : memref<f32>) <{modifiers = #acc<data_clause_modifier zero>}> -> memref<f32>

// -----

%value = memref.alloc() : memref<f32>
// expected-error @below {{invalid data clause modifiers: readonly}}
%0 = acc.create varPtr(%value : memref<f32>) <{modifiers = #acc<data_clause_modifier readonly,zero,capture,always>}> -> memref<f32>

// -----

func.func @fct1(%0 : !llvm.ptr) -> () {
  // expected-error @below {{expected symbol reference @privatization_i32 to point to a private declaration}}
  %priv = acc.private varPtr(%0 : !llvm.ptr) varType(i32) recipe(@privatization_i32) -> !llvm.ptr
  return
}

// -----

acc.private.recipe @privatization_i32 : !llvm.ptr init {
^bb0(%arg0: !llvm.ptr):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %c0, %0 : i32, !llvm.ptr
  acc.yield %0 : !llvm.ptr
}

func.func @fct1(%0 : !llvm.ptr) -> () {
  %priv = acc.private varPtr(%0 : !llvm.ptr) varType(i32) recipe(@privatization_i32) -> !llvm.ptr
  // expected-error @below {{expected firstprivate as defining op}}
  acc.serial firstprivate(%priv : !llvm.ptr) {
  }
  return
}

// -----

acc.private.recipe @privatization_i32 : !llvm.ptr init {
^bb0(%arg0: !llvm.ptr):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %c0, %0 : i32, !llvm.ptr
  acc.yield %0 : !llvm.ptr
}

func.func @fct1(%0 : !llvm.ptr) -> () {
  %priv = acc.private varPtr(%0 : !llvm.ptr) varType(i32) recipe(@privatization_i32) -> !llvm.ptr
  // expected-error @below {{op private operand appears more than once}}
  acc.serial private(%priv, %priv : !llvm.ptr, !llvm.ptr) {
  }
  return
}

// -----

func.func @fct1(%0 : !llvm.ptr) -> () {
  // expected-error @below {{op recipe expected for private}}
  %priv = acc.private varPtr(%0 : !llvm.ptr) varType(i32) -> !llvm.ptr
  return
}

// -----

func.func @fct1(%0 : !llvm.ptr) -> () {
  // expected-error @below {{op recipe expected for firstprivate}}
  %priv = acc.firstprivate varPtr(%0 : !llvm.ptr) varType(i32) -> !llvm.ptr
  return
}

// -----

func.func @fct1(%0 : !llvm.ptr) -> () {
  // expected-error @below {{op recipe expected for reduction}}
  %priv = acc.reduction varPtr(%0 : !llvm.ptr) varType(i32) -> !llvm.ptr
  return
}

// -----

func.func @verify_declare_enter(%arg0 : memref<i32>) {
// expected-error @below {{expect valid declare data entry operation or acc.getdeviceptr as defining op}}
  %0 = acc.declare_enter dataOperands(%arg0 : memref<i32>)
  acc.declare_exit token(%0) dataOperands(%arg0 : memref<i32>)
  return
}

func.func @verify_data(%arg0 : memref<i32>) {
// expected-error @below {{expect data entry/exit operation or acc.getdeviceptr as defining op}}
  acc.data dataOperands(%arg0 : memref<i32>) {
    acc.terminator
  }
  return
}

// -----

func.func @verify_host_data_duplicate_use_device(%arg0 : memref<i32>) {
  %0 = acc.use_device varPtr(%arg0 : memref<i32>) -> memref<i32>
  %1 = acc.use_device varPtr(%arg0 : memref<i32>) -> memref<i32>
// expected-error @below {{duplicate use_device variable}}
  acc.host_data dataOperands(%0, %1 : memref<i32>, memref<i32>) {
    acc.terminator
  }
  return
}

// -----

// Regression test for https://github.com/llvm/llvm-project/issues/107027.
// acc.parallel with async operands but no asyncOperandsDeviceType attribute
// must produce a diagnostic instead of crashing in verifyDeviceTypeCountMatch.

func.func @verify_parallel_async_missing_device_type(%arg0: i64) {
// expected-error @below {{async operands count must match async device_type count}}
  "acc.parallel"(%arg0) <{
    operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>
  }> ({
    acc.yield
  }) : (i64) -> ()
  return
}

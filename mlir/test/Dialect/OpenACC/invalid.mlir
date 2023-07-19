// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop gang {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop worker {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop vector {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop gang worker {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop gang vector {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop worker vector {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop gang worker vector {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{expected non-empty body.}}
acc.loop {
}

// -----

// expected-error@+1 {{only one of "auto", "independent", "seq" can be present at the same time}}
acc.loop {
  acc.yield
} attributes {auto_, seq}

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
// expected-error@+1 {{wait_devnum cannot appear without waitOperands}}
acc.update wait_devnum(%cst: index) dataOperands(%0: memref<f32>)

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<f32>
%0 = acc.update_device varPtr(%value : memref<f32>) -> memref<f32>
// expected-error@+1 {{async attribute cannot appear with asyncOperand}}
acc.update async(%cst: index) dataOperands(%0 : memref<f32>) attributes {async}

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<f32>
%0 = acc.update_device varPtr(%value : memref<f32>) -> memref<f32>
// expected-error@+1 {{wait attribute cannot appear with waitOperands}}
acc.update wait(%cst: index) dataOperands(%0: memref<f32>) attributes {wait}

// -----

%cst = arith.constant 1 : index
// expected-error@+1 {{wait_devnum cannot appear without waitOperands}}
acc.wait wait_devnum(%cst: index)

// -----

%cst = arith.constant 1 : index
// expected-error@+1 {{async attribute cannot appear with asyncOperand}}
acc.wait async(%cst: index) attributes {async}

// -----

acc.parallel {
// expected-error@+1 {{'acc.init' op cannot be nested in a compute operation}}
  acc.init
  acc.yield
}

// -----

acc.loop {
// expected-error@+1 {{'acc.init' op cannot be nested in a compute operation}}
  acc.init
  acc.yield
}

// -----

acc.parallel {
// expected-error@+1 {{'acc.shutdown' op cannot be nested in a compute operation}}
  acc.shutdown
  acc.yield
}

// -----

acc.loop {
// expected-error@+1 {{'acc.shutdown' op cannot be nested in a compute operation}}
  acc.shutdown
  acc.yield
}

// -----

acc.loop {
  "test.openacc_dummy_op"() ({
    // expected-error@+1 {{'acc.shutdown' op cannot be nested in a compute operation}}
    acc.shutdown
  }) : () -> ()
  acc.yield
}

// -----

// expected-error@+1 {{at least one operand must be present in dataOperands on the exit data operation}}
acc.exit_data attributes {async}

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<f32>
%0 = acc.getdeviceptr varPtr(%value : memref<f32>) -> memref<f32>
// expected-error@+1 {{async attribute cannot appear with asyncOperand}}
acc.exit_data async(%cst: index) dataOperands(%0 : memref<f32>) attributes {async}
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
acc.enter_data attributes {async}

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<f32>
%0 = acc.create varPtr(%value : memref<f32>) -> memref<f32>
// expected-error@+1 {{async attribute cannot appear with asyncOperand}}
acc.enter_data async(%cst: index) dataOperands(%0 : memref<f32>) attributes {async}

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<f32>
%0 = acc.create varPtr(%value : memref<f32>) -> memref<f32>
// expected-error@+1 {{wait attribute cannot appear with waitOperands}}
acc.enter_data wait(%cst: index) dataOperands(%0 : memref<f32>) attributes {wait}

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
acc.private.recipe @privatization_i32 : !llvm.ptr<i32> init {
}

// -----

// expected-error@+1 {{expects init region first argument of the privatization type}}
acc.private.recipe @privatization_i32 : !llvm.ptr<i32> init {
^bb0(%arg0 : !llvm.ptr<f32>):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<i32>
  llvm.store %c0, %0 : !llvm.ptr<i32>
  acc.yield %0 : !llvm.ptr<i32>
}

// -----

// expected-error@+1 {{expects init region to yield a value of the privatization type}}
acc.private.recipe @privatization_i32 : !llvm.ptr<f32> init {
^bb0(%arg0 : !llvm.ptr<f32>):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<i32>
  llvm.store %c0, %0 : !llvm.ptr<i32>
  acc.yield %0 : !llvm.ptr<i32>
}

// -----

// expected-error@+1 {{expects destroy region first argument of the privatization type}}
acc.private.recipe @privatization_i32 : !llvm.ptr<i32> init {
^bb0(%arg0 : !llvm.ptr<i32>):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<i32>
  llvm.store %c0, %0 : !llvm.ptr<i32>
  acc.yield %0 : !llvm.ptr<i32>
} destroy {
^bb0(%arg0 : f32):
  "test.openacc_dummy_op"(%arg0) : (f32) -> ()
}

// -----

// expected-error@+1 {{expects non-empty init region}}
acc.firstprivate.recipe @privatization_i32 : !llvm.ptr<i32> init {
} copy {}

// -----

// expected-error@+1 {{expects init region first argument of the privatization type}}
acc.firstprivate.recipe @privatization_i32 : !llvm.ptr<i32> init {
^bb0(%arg0 : !llvm.ptr<f32>):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<i32>
  llvm.store %c0, %0 : !llvm.ptr<i32>
  acc.yield %0 : !llvm.ptr<i32>
} copy {}

// -----

// expected-error@+1 {{expects init region to yield a value of the privatization type}}
acc.firstprivate.recipe @privatization_i32 : !llvm.ptr<f32> init {
^bb0(%arg0 : !llvm.ptr<f32>):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<i32>
  llvm.store %c0, %0 : !llvm.ptr<i32>
  acc.yield %0 : !llvm.ptr<i32>
} copy {}

// -----

// expected-error@+1 {{expects non-empty copy region}}
acc.firstprivate.recipe @privatization_i32 : !llvm.ptr<i32> init {
^bb0(%arg0 : !llvm.ptr<i32>):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<i32>
  llvm.store %c0, %0 : !llvm.ptr<i32>
  acc.yield %0 : !llvm.ptr<i32>
} copy {
}

// -----

// expected-error@+1 {{expects copy region with two arguments of the privatization type}}
acc.firstprivate.recipe @privatization_i32 : !llvm.ptr<i32> init {
^bb0(%arg0 : !llvm.ptr<i32>):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<i32>
  llvm.store %c0, %0 : !llvm.ptr<i32>
  acc.yield %0 : !llvm.ptr<i32>
} copy {
^bb0(%arg0 : f32):
  "test.openacc_dummy_op"(%arg0) : (f32) -> ()
}

// -----

// expected-error@+1 {{expects copy region with two arguments of the privatization type}}
acc.firstprivate.recipe @privatization_i32 : !llvm.ptr<i32> init {
^bb0(%arg0 : !llvm.ptr<i32>):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<i32>
  llvm.store %c0, %0 : !llvm.ptr<i32>
  acc.yield %0 : !llvm.ptr<i32>
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
^bb0(%arg0 : i32, %arg1 : !llvm.ptr<i32>):
  llvm.store %arg0, %arg1 : !llvm.ptr<i32>
  acc.yield
} destroy {
^bb0(%arg0 : f32):
  acc.yield
}

// -----

// expected-error@+1 {{expected ')'}}
acc.loop gang(static=%i64Value: i64, num=%i64Value: i64 {
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

// expected-error@+1 {{new value expected after comma}}
acc.loop gang(static=%i64Value: i64, ) {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
}

// -----

func.func @fct1(%0 : !llvm.ptr<i32>) -> () {
  // expected-error@+1 {{expected symbol reference @privatization_i32 to point to a private declaration}}
  acc.serial private(@privatization_i32 -> %0 : !llvm.ptr<i32>) {
  }
  return
}

// -----

// expected-error@+1 {{expect at least one of num, dim or static values}}
acc.loop gang() {
  "test.openacc_dummy_op"() : () -> ()
  acc.yield
}

// -----

%i64value = arith.constant 1 : i64
// expected-error@+1 {{num_gangs expects a maximum of 3 values}}
acc.parallel num_gangs(%i64value, %i64value, %i64value, %i64value : i64, i64, i64, i64) {
}

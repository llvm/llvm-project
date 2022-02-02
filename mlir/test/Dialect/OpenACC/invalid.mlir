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

// expected-error@+1 {{only one of auto, independent, seq can be present at the same time}}
acc.loop {
  acc.yield
} attributes {auto_, seq}

// -----

// expected-error@+1 {{at least one operand or the default attribute must appear on the data operation}}
acc.data {
  acc.yield
}

// -----

// expected-error@+1 {{at least one value must be present in hostOperands or deviceOperands}}
acc.update

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{wait_devnum cannot appear without waitOperands}}
acc.update wait_devnum(%cst: index) host(%value: memref<10xf32>)

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{async attribute cannot appear with asyncOperand}}
acc.update async(%cst: index) host(%value: memref<10xf32>) attributes {async}

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{wait attribute cannot appear with waitOperands}}
acc.update wait(%cst: index) host(%value: memref<10xf32>) attributes {wait}

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

// expected-error@+1 {{at least one operand in copyout, delete or detach must appear on the exit data operation}}
acc.exit_data attributes {async}

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{async attribute cannot appear with asyncOperand}}
acc.exit_data async(%cst: index) delete(%value : memref<10xf32>) attributes {async}

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{wait_devnum cannot appear without waitOperands}}
acc.exit_data wait_devnum(%cst: index) delete(%value : memref<10xf32>)

// -----

// expected-error@+1 {{at least one operand in copyin, create, create_zero or attach must appear on the enter data operation}}
acc.enter_data attributes {async}

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{async attribute cannot appear with asyncOperand}}
acc.enter_data async(%cst: index) create(%value : memref<10xf32>) attributes {async}

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{wait attribute cannot appear with waitOperands}}
acc.enter_data wait(%cst: index) create(%value : memref<10xf32>) attributes {wait}

// -----

%cst = arith.constant 1 : index
%value = memref.alloc() : memref<10xf32>
// expected-error@+1 {{wait_devnum cannot appear without waitOperands}}
acc.enter_data wait_devnum(%cst: index) create(%value : memref<10xf32>)

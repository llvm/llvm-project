// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop gang {
  "some.op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop worker {
  "some.op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop vector {
  "some.op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop gang worker {
  "some.op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop gang vector {
  "some.op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop worker vector {
  "some.op"() : () -> ()
  acc.yield
} attributes {seq}

// -----

// expected-error@+1 {{gang, worker or vector cannot appear with the seq attr}}
acc.loop gang worker vector {
  "some.op"() : () -> ()
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

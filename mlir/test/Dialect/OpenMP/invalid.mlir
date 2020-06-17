// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func @unknown_clause() {
  // expected-error@+1 {{invalid is not a valid clause for the omp.parallel operation}}
  omp.parallel invalid {
  }

  return
}

// -----

func @if_once(%n : i1) {
  // expected-error@+1 {{at most one if clause can appear on the omp.parallel operation}}
  omp.parallel if(%n) if(%n) {
  }

  return
}

// -----

func @num_threads_once(%n : si32) {
  // expected-error@+1 {{at most one num_threads clause can appear on the omp.parallel operation}}
  omp.parallel num_threads(%n : si32) num_threads(%n : si32) {
  }

  return
}

// -----

func @private_once(%n : memref<i32>) {
  // expected-error@+1 {{at most one private clause can appear on the omp.parallel operation}}
  omp.parallel private(%n : memref<i32>) private(%n : memref<i32>) {
  }

  return
}

// -----

func @firstprivate_once(%n : memref<i32>) {
  // expected-error@+1 {{at most one firstprivate clause can appear on the omp.parallel operation}}
  omp.parallel firstprivate(%n : memref<i32>) firstprivate(%n : memref<i32>) {
  }

  return
}

// -----

func @shared_once(%n : memref<i32>) {
  // expected-error@+1 {{at most one shared clause can appear on the omp.parallel operation}}
  omp.parallel shared(%n : memref<i32>) shared(%n : memref<i32>) {
  }

  return
}

// -----

func @copyin_once(%n : memref<i32>) {
  // expected-error@+1 {{at most one copyin clause can appear on the omp.parallel operation}}
  omp.parallel copyin(%n : memref<i32>) copyin(%n : memref<i32>) {
  }

  return
}

// -----
 
func @default_once() {
  // expected-error@+1 {{at most one default clause can appear on the omp.parallel operation}}
  omp.parallel default(private) default(firstprivate) {
  }

  return
}

// -----

func @proc_bind_once() {
  // expected-error@+1 {{at most one proc_bind clause can appear on the omp.parallel operation}}
  omp.parallel proc_bind(close) proc_bind(spread) {
  }

  return
}

// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @nested_unnamed_critical() {
  omp.critical {
    // expected-error @below {{cannot be nested inside another unnamed omp.critical region}}
    omp.critical {
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

omp.critical.declare @my_mutex

func.func @nested_named_critical() {
  omp.critical(@my_mutex) {
    // expected-error @below {{cannot be nested inside another omp.critical region with the same name (@my_mutex)}}
    omp.critical(@my_mutex) {
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

omp.critical.declare @my_mutex

func.func @nested_named_critical_indirect() {
  omp.critical(@my_mutex) {
    omp.single {
      // expected-error @below {{cannot be nested inside another omp.critical region with the same name (@my_mutex)}}
      omp.critical(@my_mutex) {
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

omp.critical.declare @my_mutex_A
omp.critical.declare @my_mutex_B

func.func @nested_named_critical_interleaved() {
  omp.critical(@my_mutex_A) {
    omp.critical(@my_mutex_B) {
      // expected-error @below {{cannot be nested inside another omp.critical region with the same name (@my_mutex_A)}}
      omp.critical(@my_mutex_A) {
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

omp.critical.declare @my_mutex_outer
omp.critical.declare @my_mutex_inner

func.func @nested_critical_different_names() {
  omp.critical(@my_mutex_outer) {
    // Valid: Names are different.
    omp.critical(@my_mutex_inner) {
      omp.terminator
    }
    omp.terminator
  }
  return
}

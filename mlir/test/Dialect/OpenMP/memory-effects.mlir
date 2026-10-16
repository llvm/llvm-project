// RUN: mlir-opt %s --test-side-effects --verify-diagnostics

// Check the trait on omp.masked itself, independently of its body.
func.func @masked_recursive_memory_effects() {
  // expected-remark@+1 {{recursive memory effects}}
  omp.masked {
    omp.terminator
  }
  return
}

// Check the trait on omp.master itself, independently of its body.
func.func @master_recursive_memory_effects() {
  // expected-remark@+1 {{recursive memory effects}}
  omp.master {
    omp.terminator
  }
  return
}

func.func @sections_effects() {
  // An implicit barrier reads and writes unspecified memory.
  // expected-remark@+2 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on resource '<Default>'}}
  omp.sections {
    omp.section {
      omp.terminator
    }
    omp.terminator
  }
  return
}

func.func @sections_nowait_effects() {
  // Without a reduction clause, nowait removes the implicit barrier effects.
  // expected-remark@+1 {{operation has no memory effects}}
  omp.sections nowait {
    omp.section {
      omp.terminator
    }
    omp.terminator
  }
  return
}

omp.private {type = firstprivate} @private_i32 : memref<i32> copy {
^bb0(%original: memref<i32>, %private: memref<i32>):
  // expected-remark@+1 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  %v = memref.load %original[] : memref<i32>
  // expected-remark@+1 {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  memref.store %v, %private[] : memref<i32>
  omp.yield(%private : memref<i32>)
}
omp.private {type = private} @private_ptr : !llvm.ptr

func.func @sections_private_effects(%x: memref<i32>, %y: !llvm.ptr) {
  // expected-remark@below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'allocate' on block argument 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'write' on block argument 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'free' on block argument 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'read' on op operand 1, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'allocate' on block argument 1, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'write' on block argument 1, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'free' on block argument 1, on resource '<Default>'}}
  omp.sections nowait private(@private_i32 %x -> %px,
      @private_ptr %y -> %py : memref<i32>, !llvm.ptr) {
    omp.section {
    ^bb0(%sx: memref<i32>, %sy: !llvm.ptr):
      omp.terminator
    }
    omp.terminator
  } <{assume_bounded_recipe_effects}>
  return
}

omp.declare_reduction @add_f32 : f32
init {
^bb0(%unused: f32):
  // expected-remark@+1 {{operation has no memory effects}}
  %zero = arith.constant 0.0 : f32
  omp.yield(%zero : f32)
}
combiner {
^bb0(%lhs: f32, %rhs: f32):
  // expected-remark@+1 {{operation has no memory effects}}
  %sum = arith.addf %lhs, %rhs : f32
  omp.yield(%sum : f32)
}

func.func @sections_reduction_effects(%x: memref<f32>) {
  // TODO: A nowait clause does not currently eliminate a barrier from a sections construct.
  // The first two effects come from the implicit barrier.
  // expected-remark@below {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'write' on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'allocate' on block argument 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'free' on block argument 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'read' on block argument 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'write' on block argument 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'write' on op operand 0, on resource '<Default>'}}
  omp.sections nowait reduction(
      @add_f32 %x -> %private_x : memref<f32>) {
    omp.section {
    ^bb0(%section_x: memref<f32>):
      omp.terminator
    }
    omp.terminator
  } <{assume_bounded_recipe_effects}>
  return
}

func.func private @custom_combiner(f32, f32) -> f32

omp.declare_reduction @custom_f32 : f32
init {
^bb0(%mold: f32):
  omp.yield(%mold : f32)
}
combiner {
^bb0(%lhs: f32, %rhs: f32):
  %v = func.call @custom_combiner(%lhs, %rhs) : (f32, f32) -> f32
  omp.yield(%v : f32)
}

func.func @sections_custom_scalar_reduction(%x: memref<f32>) {
  // A custom scalar reduction can access arbitrary memory through a call.
  // expected-remark@below {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'write' on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'allocate' on block argument 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'free' on block argument 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'read' on block argument 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'write' on block argument 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'write' on op operand 0, on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'allocate' on resource '<Default>'}}
  // expected-remark@below {{found an instance of 'free' on resource '<Default>'}}
  omp.sections nowait reduction(@custom_f32 %x -> %p : memref<f32>) {
    omp.section {
    ^bb0(%s: memref<f32>):
      omp.terminator
    }
    omp.terminator
  }
  return
}

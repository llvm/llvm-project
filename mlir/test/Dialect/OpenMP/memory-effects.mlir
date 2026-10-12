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

omp.private {type = private} @sections_p : memref<i32>

func.func @sections_private_implicit_allocation(%x: memref<i32>) {
  // The directive allocates private storage even though the recipe is empty.
  // expected-remark@+1 {{found an instance of 'allocate' on block argument 0, on resource '<Default>'}}
  omp.sections nowait private(@sections_p %x -> %private_x : memref<i32>) {
    omp.section {
    ^bb0(%section_x: memref<i32>):
      omp.terminator
    }
    omp.terminator
  }
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
  // The first two effects come from the implicit barrier, which nowait does
  // not currently eliminate when a reduction clause is present. Allocation,
  // initialization and combination access memory outside the by-value recipe.
  // expected-remark@+7 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+6 {{found an instance of 'write' on resource '<Default>'}}
  // expected-remark@+5 {{found an instance of 'allocate' on block argument 0, on resource '<Default>'}}
  // expected-remark@+4 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark@+3 {{found an instance of 'write' on block argument 0, on resource '<Default>'}}
  // expected-remark@+2 {{found an instance of 'read' on block argument 0, on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on op operand 0, on resource '<Default>'}}
  omp.sections nowait reduction(
      @add_f32 %x -> %private_x : memref<f32>) {
    omp.section {
    ^bb0(%section_x: memref<f32>):
      omp.terminator
    }
    omp.terminator
  }
  return
}

// Effects nested in the copy region must be traced through subviews to the
// original mold and private arguments.
omp.private {type = firstprivate} @sections_copy_subviews : memref<4xi32> copy {
^bb0(%mold: memref<4xi32>, %private: memref<4xi32>):
  // Use scf.execute_region to test recursive effects.
  scf.execute_region {
    // expected-remark@+1 {{operation has no memory effects}}
    %c0 = arith.constant 0 : index
    // expected-remark@+1 {{operation has no memory effects}}
    %mold_view = memref.subview %mold[1] [2] [1] : memref<4xi32> to memref<2xi32, strided<[1], offset: 1>>
    // expected-remark@+1 {{operation has no memory effects}}
    %private_view = memref.subview %private[1] [2] [1] : memref<4xi32> to memref<2xi32, strided<[1], offset: 1>>
    // expected-remark@+1 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
    %value = memref.load %mold_view[%c0] : memref<2xi32, strided<[1], offset: 1>>
    // expected-remark@+1 {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
    memref.store %value, %private_view[%c0] : memref<2xi32, strided<[1], offset: 1>>
    scf.yield
  }
  omp.yield(%private : memref<4xi32>)
} dealloc {
^bb0(%private: memref<4xi32>):
  // expected-remark@+1 {{found an instance of 'free' on op operand 0, on resource '<Default>'}}
  memref.dealloc %private : memref<4xi32>
  omp.yield
}

func.func @sections_private_subview_effects(%x: memref<4xi32>) {
  // expected-remark@+4 {{found an instance of 'allocate' on block argument 0, on resource '<Default>'}}
  // expected-remark@+3 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark@+2 {{found an instance of 'write' on block argument 0, on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'free' on block argument 0, on resource '<Default>'}}
  omp.sections nowait private(@sections_copy_subviews %x -> %private_x : memref<4xi32>) {
    omp.section {
    ^bb0(%section_x: memref<4xi32>):
      omp.terminator
    }
    omp.terminator
  }
  return
}

// Model allocation, initialisation, combination and cleanup effects. 
// The allocation is local to the recipe initialiser.
// The combiner LHS can be the source or a private accumulator.
omp.declare_reduction @sections_add_byref : memref<f32>
alloc {
  // expected-remark@+1 {{found an instance of 'allocate' on op result 0, on resource '<Default>'}}
  %allocated = memref.alloc() : memref<f32>
  omp.yield(%allocated : memref<f32>)
}
init {
^bb0(%mold: memref<f32>, %private: memref<f32>):
  // expected-remark@+1 {{operation has no memory effects}}
  %zero = arith.constant 0.0 : f32
  // expected-remark@+1 {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  memref.store %zero, %private[] : memref<f32>
  omp.yield(%private : memref<f32>)
}
combiner {
^bb0(%lhs: memref<f32>, %rhs: memref<f32>):
  // expected-remark@+1 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  %left = memref.load %lhs[] : memref<f32>
  // expected-remark@+1 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  %right = memref.load %rhs[] : memref<f32>
  // expected-remark@+1 {{operation has no memory effects}}
  %sum = arith.addf %left, %right : f32
  // expected-remark@+1 {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  memref.store %sum, %lhs[] : memref<f32>
  omp.yield(%lhs : memref<f32>)
}
cleanup {
^bb0(%private: memref<f32>):
  // expected-remark@+1 {{found an instance of 'free' on op operand 0, on resource '<Default>'}}
  memref.dealloc %private : memref<f32>
  omp.yield
}

func.func @sections_reduction_recipe_effects(%x: memref<f32>) {
  // The first two effects come from the implicit barrier.
  // expected-remark@+8 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+7 {{found an instance of 'write' on resource '<Default>'}}
  // expected-remark@+6 {{found an instance of 'allocate' on resource '<Default>'}}
  // expected-remark@+5 {{found an instance of 'write' on block argument 0, on resource '<Default>'}}
  // expected-remark@+4 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark@+3 {{found an instance of 'read' on block argument 0, on resource '<Default>'}}
  // expected-remark@+2 {{found an instance of 'write' on op operand 0, on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'free' on block argument 0, on resource '<Default>'}}
  omp.sections nowait reduction(byref @sections_add_byref %x -> %private_x : memref<f32>) {
    omp.section {
    ^bb0(%section_x: memref<f32>):
      omp.terminator
    }
    omp.terminator
  }
  return
}

// Unknown effects inside a nested region require an untargeted read and write,
// while retaining known effects already collected from the recipe.
func.func private @unknown_recipe_effects(memref<i32>, i32)

omp.private {type = firstprivate} @sections_unknown_copy : memref<i32> copy {
^bb0(%mold: memref<i32>, %private: memref<i32>):
  // expected-remark@+1 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  %value = memref.load %mold[] : memref<i32>
  scf.execute_region {
    func.call @unknown_recipe_effects(%private, %value) : (memref<i32>, i32) -> ()
    scf.yield
  }
  omp.yield(%private : memref<i32>)
}

func.func @sections_private_unknown_recipe_effects(%x: memref<i32>) {
  // expected-remark@+6 {{found an instance of 'allocate' on block argument 0, on resource '<Default>'}}
  // expected-remark@+5 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark@+4 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+3 {{found an instance of 'write' on resource '<Default>'}}
  // expected-remark@+2 {{found an instance of 'allocate' on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'free' on resource '<Default>'}}
  omp.sections nowait private(@sections_unknown_copy %x -> %private_x : memref<i32>) {
    omp.section {
    ^bb0(%section_x: memref<i32>):
      omp.terminator
    }
    omp.terminator
  }
  return
}

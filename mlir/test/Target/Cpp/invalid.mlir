// RUN: mlir-translate -split-input-file -mlir-to-cpp -verify-diagnostics %s

// expected-error@+1 {{'func.func' op with multiple blocks needs variables declared at top}}
func.func @multiple_blocks() {
^bb1:
    cf.br ^bb2
^bb2:
    return
}

// -----

func.func @unsupported_op(%arg0: i1) {
  // expected-error@+1 {{'cf.assert' op unable to find printer for op}}
  cf.assert %arg0, "assertion foo"
  return
}

// -----

// expected-error@+1 {{cannot emit integer type 'i80'}}
func.func @unsupported_integer_type(%arg0 : i80) {
  return
}

// -----

// expected-error@+1 {{cannot emit float type 'f80'}}
func.func @unsupported_float_type(%arg0 : f80) {
  return
}

// -----

// expected-error@+1 {{cannot emit type 'memref<100xf32>'}}
func.func @memref_type(%arg0 : memref<100xf32>) {
  return
}

// -----

// expected-error@+1 {{cannot emit type 'vector<100xf32>'}}
func.func @vector_type(%arg0 : vector<100xf32>) {
  return
}

// -----

// expected-error@+1 {{cannot emit tensor type with non static shape}}
func.func @non_static_shape(%arg0 : tensor<?xf32>) {
  return
}

// -----

// expected-error@+1 {{cannot emit unranked tensor type}}
func.func @unranked_tensor(%arg0 : tensor<*xf32>) {
  return
}

// -----

// expected-error@+1 {{cannot emit tensor of array type}}
func.func @tensor_of_array(%arg0 : tensor<4x!emitc.array<4xf32>>) {
  return
}

// -----

// expected-error@+1 {{cannot emit pointer to array type}}
func.func @pointer_to_array(%arg0 : !emitc.ptr<!emitc.array<4xf32>>) {
  return
}

// -----

// expected-error@+1 {{cannot emit array type as result type}}
func.func @array_as_result(%arg: !emitc.array<4xi8>) -> (!emitc.array<4xi8>) {
   return %arg : !emitc.array<4xi8>
}

// -----

func.func @ptr_to_array() {
  // expected-error@+1 {{cannot emit pointer to array type '!emitc.ptr<!emitc.array<9xi16>>'}}
  %v = "emitc.variable"(){value = #emitc.opaque<"NULL">} : () -> !emitc.ptr<!emitc.array<9xi16>>
  return
}

// -----

func.func @emitc_switch() {
  %0 = "emitc.variable"(){value = 1 : ui16} : () -> ui16

  // expected-error@+1 {{'emitc.switch' op expected region to end with emitc.yield, but got emitc.call_opaque}}
  emitc.switch %0 : ui16
  case 2: {
    %1 = emitc.call_opaque "func_b" () : () -> i32
  }
  case 5: {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default: {
    %3 = "emitc.variable"(){value = 42.0 : f32} : () -> f32
    %4 = "emitc.variable"(){value = 42.0 : f32} : () -> f32

    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.call_opaque "func3" (%3, %4) { args = [1 : index, 0 : index] } : (f32, f32) -> ()
    emitc.yield
  }
  return
}

// -----

func.func @emitc_switch() {
  %0 = "emitc.variable"(){value = 1 : ui16} : () -> ui16

  // expected-error@+1 {{'emitc.switch' op expected region to end with emitc.yield, but got emitc.call_opaque}}
  emitc.switch %0 : ui16
  case 2: {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5: {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default: {
    %3 = "emitc.variable"(){value = 42.0 : f32} : () -> f32
    %4 = "emitc.variable"(){value = 42.0 : f32} : () -> f32

    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.call_opaque "func3" (%3, %4) { args = [1 : index, 0 : index] } : (f32, f32) -> ()
  }
  return
}

// -----

func.func @emitc_switch() {
  %0 = "emitc.variable"(){value = 1 : i16} : () -> i16

  emitc.switch %0 : i16
  case 2: {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  // expected-error@+1 {{custom op 'emitc.switch' expected integer value}}
  case : {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default: {
    %3 = "emitc.variable"(){value = 42.0 : f32} : () -> f32
    %4 = "emitc.variable"(){value = 42.0 : f32} : () -> f32

    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.call_opaque "func3" (%3, %4) { args = [1 : index, 0 : index] } : (f32, f32) -> ()
    emitc.yield
  }
  return
}

// -----

func.func @emitc_switch() {
  %0 = "emitc.variable"(){value = 1 : i16} : () -> i16

  emitc.switch %0 : i16
  case 2: {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 3: {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  // expected-error@+1 {{custom op 'emitc.switch' expected 'default'}}
  return
}

// -----

func.func @emitc_switch() {
  %0 = "emitc.variable"(){value = 1 : i16} : () -> i16

  // expected-error@+1 {{'emitc.switch' op has duplicate case value: 2}}
  emitc.switch %0 : i16
  case 2: {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 2: {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default: {
    %3 = "emitc.variable"(){value = 42.0 : f32} : () -> f32
    %4 = "emitc.variable"(){value = 42.0 : f32} : () -> f32

    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.call_opaque "func3" (%3, %4) { args = [1 : index, 0 : index] } : (f32, f32) -> ()
    emitc.yield
  }
  return
}

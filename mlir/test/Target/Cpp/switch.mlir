// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

// CHECK-LABEL:   int8_t v1 = 1;
// CHECK-LABEL:   switch(v1) {
// CHECK:         case (2): {
// CHECK:           int32_t v2 = func_b();
// CHECK:           break;
// CHECK:         }
// CHECK:         case (5): {
// CHECK:           int32_t v3 = func_a();
// CHECK:           break;
// CHECK:         }
// CHECK:         default: {
// CHECK:           float v4 = 4.200000000e+01f;
// CHECK:           float v5 = 4.200000000e+01f;
// CHECK:           func2(v4);
// CHECK:           func3(v5, v4);
// CHECK:           break;
// CHECK:         }
// CHECK:         return;
// CHECK:       }
func.func @emitc_switch_i8() {
  %0 = "emitc.variable"(){value = 1 : i8} : () -> i8

  emitc.switch %0 : i8
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
    emitc.yield
  }
  return
}


// CHECK-LABEL: void emitc_switch_ui8() {
// CHECK:         uint8_t v1 = 1;
// CHECK:         switch(v1) {
// CHECK:         case (2): {
// CHECK:           int32_t v2 = func_b();
// CHECK:           break;
// CHECK:         }
// CHECK:         case (5): {
// CHECK:           int32_t v3 = func_a();
// CHECK:           break;
// CHECK:         }
// CHECK:         default: {
// CHECK:           float v4 = 4.200000000e+01f;
// CHECK:           float v5 = 4.200000000e+01f;
// CHECK:           func2(v4);
// CHECK:           func3(v5, v4);
// CHECK:           break;
// CHECK:         }
// CHECK:         return;
// CHECK:       }
func.func @emitc_switch_ui8() {
  %0 = "emitc.variable"(){value = 1 : ui8} : () -> ui8

  emitc.switch %0 : ui8
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
    emitc.yield
  }
  return
}

// CHECK-LABEL: void emitc_switch_i16() {
// CHECK:         int16_t v1 = 1;
// CHECK:         switch(v1) {
// CHECK:         case (2): {
// CHECK:           int32_t v2 = func_b();
// CHECK:           break;
// CHECK:         }
// CHECK:         case (5): {
// CHECK:           int32_t v3 = func_a();
// CHECK:           break;
// CHECK:         }
// CHECK:         default: {
// CHECK:           float v4 = 4.200000000e+01f;
// CHECK:           float v5 = 4.200000000e+01f;
// CHECK:           func2(v4);
// CHECK:           func3(v5, v4);
// CHECK:           break;
// CHECK:         }
// CHECK:         return;
// CHECK:       }
func.func @emitc_switch_i16() {
  %0 = "emitc.variable"(){value = 1 : i16} : () -> i16

  emitc.switch %0 : i16
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
    emitc.yield
  }
  return
}

// CHECK-LABEL: void emitc_switch_ui16() {
// CHECK:         uint16_t v1 = 1;
// CHECK:         switch(v1) {
// CHECK:         case (2): {
// CHECK:           int32_t v2 = func_b();
// CHECK:           break;
// CHECK:         }
// CHECK:         case (5): {
// CHECK:           int32_t v3 = func_a();
// CHECK:           break;
// CHECK:         }
// CHECK:         default: {
// CHECK:           float v4 = 4.200000000e+01f;
// CHECK:           float v5 = 4.200000000e+01f;
// CHECK:           func2(v4);
// CHECK:           func3(v5, v4);
// CHECK:           break;
// CHECK:         }
// CHECK:         return;
// CHECK:       }
func.func @emitc_switch_ui16() {
  %0 = "emitc.variable"(){value = 1 : ui16} : () -> ui16

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
    emitc.yield
  }
  return
}

// CHECK-LABEL: void emitc_switch_i32() {
// CHECK:         int32_t v1 = 1;
// CHECK:         switch(v1) {
// CHECK:         case (2): {
// CHECK:           int32_t v2 = func_b();
// CHECK:           break;
// CHECK:         }
// CHECK:         case (5): {
// CHECK:           int32_t v3 = func_a();
// CHECK:           break;
// CHECK:         }
// CHECK:         default: {
// CHECK:           float v4 = 4.200000000e+01f;
// CHECK:           float v5 = 4.200000000e+01f;
// CHECK:           func2(v4);
// CHECK:           func3(v5, v4);
// CHECK:           break;
// CHECK:         }
// CHECK:         return;
// CHECK:       }
func.func @emitc_switch_i32() {
  %0 = "emitc.variable"(){value = 1 : i32} : () -> i32

  emitc.switch %0 : i32
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
    emitc.yield
  }
  return
}

// CHECK-LABEL: void emitc_switch_ui32() {
// CHECK:         uint32_t v1 = 1;
// CHECK:         switch(v1) {
// CHECK:         case (2): {
// CHECK:           int32_t v2 = func_b();
// CHECK:           break;
// CHECK:         }
// CHECK:         case (5): {
// CHECK:           int32_t v3 = func_a();
// CHECK:           break;
// CHECK:         }
// CHECK:         default: {
// CHECK:           float v4 = 4.200000000e+01f;
// CHECK:           float v5 = 4.200000000e+01f;
// CHECK:           func2(v4);
// CHECK:           func3(v5, v4);
// CHECK:           break;
// CHECK:         }
// CHECK:         return;
// CHECK:       }
func.func @emitc_switch_ui32() {
  %0 = "emitc.variable"(){value = 1 : ui32} : () -> ui32

  emitc.switch %0 : ui32
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
    emitc.yield
  }
  return
}

// CHECK-LABEL: void emitc_switch_i64() {
// CHECK:         int64_t v1 = 1;
// CHECK:         switch(v1) {
// CHECK:         case (2): {
// CHECK:           int32_t v2 = func_b();
// CHECK:           break;
// CHECK:         }
// CHECK:         case (5): {
// CHECK:           int32_t v3 = func_a();
// CHECK:           break;
// CHECK:         }
// CHECK:         default: {
// CHECK:           float v4 = 4.200000000e+01f;
// CHECK:           float v5 = 4.200000000e+01f;
// CHECK:           func2(v4);
// CHECK:           func3(v5, v4);
// CHECK:           break;
// CHECK:         }
// CHECK:         return;
// CHECK:       }
func.func @emitc_switch_i64() {
  %0 = "emitc.variable"(){value = 1 : i64} : () -> i64

  emitc.switch %0 : i64
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
    emitc.yield
  }
  return
}

// CHECK-LABEL: void emitc_switch_ui64() {
// CHECK:         uint64_t v1 = 1;
// CHECK:         switch(v1) {
// CHECK:         case (2): {
// CHECK:           int32_t v2 = func_b();
// CHECK:           break;
// CHECK:         }
// CHECK:         case (5): {
// CHECK:           int32_t v3 = func_a();
// CHECK:           break;
// CHECK:         }
// CHECK:         default: {
// CHECK:           float v4 = 4.200000000e+01f;
// CHECK:           float v5 = 4.200000000e+01f;
// CHECK:           func2(v4);
// CHECK:           func3(v5, v4);
// CHECK:           break;
// CHECK:         }
// CHECK:         return;
// CHECK:       }
func.func @emitc_switch_ui64() {
  %0 = "emitc.variable"(){value = 1 : ui64} : () -> ui64

  emitc.switch %0 : ui64
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
    emitc.yield
  }
  return
}

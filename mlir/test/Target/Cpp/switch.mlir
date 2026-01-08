// RUN: mlir-translate -mlir-to-cpp %s | FileCheck --match-full-lines %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck --match-full-lines %s -check-prefix=CPP-DECLTOP

// CPP-DEFAULT-LABEL: void emitc_switch_ptrdiff_t() {
// CPP-DEFAULT:         ptrdiff_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_ptrdiff_t() {
// CPP-DECLTOP:         ptrdiff_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_ptrdiff_t() {
  %0 = "emitc.constant"(){value = 1 : index} : () -> !emitc.ptrdiff_t

  emitc.switch %0 : !emitc.ptrdiff_t
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_ssize_t() {
// CPP-DEFAULT:         ssize_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_ssize_t() {
// CPP-DECLTOP:         ssize_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_ssize_t() {
  %0 = "emitc.constant"(){value = 1 : index} : () -> !emitc.ssize_t

  emitc.switch %0 : !emitc.ssize_t
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_size_t() {
// CPP-DEFAULT:         size_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_size_t() {
// CPP-DECLTOP:         size_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_size_t() {
  %0 = "emitc.constant"(){value = 1 : index} : () -> !emitc.size_t

  emitc.switch %0 : !emitc.size_t
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_index() {
// CPP-DEFAULT:         size_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_index() {
// CPP-DECLTOP:         size_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_index() {
  %0 = "emitc.constant"(){value = 1 : index} : () -> index

  emitc.switch %0 : index
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_opaque() {
// CPP-DEFAULT:         size_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_opaque() {
// CPP-DECLTOP:         size_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_opaque() {
  %0 = "emitc.constant"() {value = #emitc.opaque<"1">} 
  : () -> !emitc.opaque<"size_t">

  emitc.switch %0 : !emitc.opaque<"size_t">
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_i1() {
// CPP-DEFAULT:         bool v1 = true;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_i1() {
// CPP-DECLTOP:         bool v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = true;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_i1() {
  %0 = "emitc.constant"(){value = 1 : i1} : () -> i1

  emitc.switch %0 : i1
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_i8() {
// CPP-DEFAULT:         int8_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_i8() {
// CPP-DECLTOP:         int8_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_i8() {
  %0 = "emitc.constant"(){value = 1 : i8} : () -> i8

  emitc.switch %0 : i8
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_ui8() {
// CPP-DEFAULT:         uint8_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_ui8() {
// CPP-DECLTOP:         uint8_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_ui8() {
  %0 = "emitc.constant"(){value = 1 : ui8} : () -> ui8

  emitc.switch %0 : ui8
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_i16() {
// CPP-DEFAULT:         int16_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_i16() {
// CPP-DECLTOP:         int16_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_i16() {
  %0 = "emitc.constant"(){value = 1 : i16} : () -> i16

  emitc.switch %0 : i16
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_ui16() {
// CPP-DEFAULT:         uint16_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_ui16() {
// CPP-DECLTOP:         uint16_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_ui16() {
  %0 = "emitc.constant"(){value = 1 : ui16} : () -> ui16

  emitc.switch %0 : ui16
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_i32() {
// CPP-DEFAULT:         int32_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_i32() {
// CPP-DECLTOP:         int32_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_i32() {
  %0 = "emitc.constant"(){value = 1 : i32} : () -> i32

  emitc.switch %0 : i32
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_ui32() {
// CPP-DEFAULT:         uint32_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_ui32() {
// CPP-DECLTOP:         uint32_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_ui32() {
  %0 = "emitc.constant"(){value = 1 : ui32} : () -> ui32

  emitc.switch %0 : ui32
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_i64() {
// CPP-DEFAULT:         int64_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_i64() {
// CPP-DECLTOP:         int64_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_i64() {
  %0 = "emitc.constant"(){value = 1 : i64} : () -> i64

  emitc.switch %0 : i64
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_ui64() {
// CPP-DEFAULT:         uint64_t v1 = 1;
// CPP-DEFAULT:         switch (v1) {
// CPP-DEFAULT:         case 2: {
// CPP-DEFAULT:           int32_t v2 = func_b();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         case 5: {
// CPP-DEFAULT:           int32_t v3 = func_a();
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           float v4 = 4.200000000e+01f;
// CPP-DEFAULT:           func2(v4);
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_ui64() {
// CPP-DECLTOP:         uint64_t v1;
// CPP-DECLTOP:         float v2;
// CPP-DECLTOP:         int32_t v3;
// CPP-DECLTOP:         int32_t v4;
// CPP-DECLTOP:         v1 = 1;
// CPP-DECLTOP:         switch (v1) {
// CPP-DECLTOP:         case 2: {
// CPP-DECLTOP:           v3 = func_b();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         case 5: {
// CPP-DECLTOP:           v4 = func_a();
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           v2 = 4.200000000e+01f;
// CPP-DECLTOP:           func2(v2);
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }
func.func @emitc_switch_ui64() {
  %0 = "emitc.constant"(){value = 1 : ui64} : () -> ui64

  emitc.switch %0 : ui64
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// CPP-DEFAULT-LABEL: void emitc_switch_expression() {
// CPP-DEFAULT:         int64_t v1 = 42;
// CPP-DEFAULT:         switch (-v1) {
// CPP-DEFAULT:         default: {
// CPP-DEFAULT:           break;
// CPP-DEFAULT:         }
// CPP-DEFAULT:         }
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

// CPP-DECLTOP-LABEL: void emitc_switch_expression() {
// CPP-DECLTOP:         int64_t v1;
// CPP-DECLTOP:         v1 = 42;
// CPP-DECLTOP:         switch (-v1) {
// CPP-DECLTOP:         default: {
// CPP-DECLTOP:           break;
// CPP-DECLTOP:         }
// CPP-DECLTOP:         }
// CPP-DECLTOP:         return;
// CPP-DECLTOP:       }

func.func @emitc_switch_expression() {
  %x = "emitc.constant"(){value = 42 : i64} : () -> i64

  %0 = emitc.expression %x : (i64) -> i64 {
    %a = emitc.unary_minus %x : (i64) -> i64
    emitc.yield %a : i64
  }

  emitc.switch %0 : i64
  default {
    emitc.yield
  }
  return
}

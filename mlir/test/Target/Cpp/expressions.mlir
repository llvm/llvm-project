// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

// CPP-DEFAULT:      int32_t single_use(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]], int32_t [[VAL_4:v[0-9]+]]) {
// CPP-DEFAULT-NEXT:   bool [[VAL_5:v[0-9]+]] = bar([[VAL_1]] * M_PI, [[VAL_3]]) - [[VAL_4]] < [[VAL_2]];
// CPP-DEFAULT-NEXT:   int32_t [[VAL_6:v[0-9]+]];
// CPP-DEFAULT-NEXT:   if ([[VAL_5]]) {
// CPP-DEFAULT-NEXT:     [[VAL_6]] = [[VAL_1]];
// CPP-DEFAULT-NEXT:   } else {
// CPP-DEFAULT-NEXT:     [[VAL_6]] = [[VAL_1]];
// CPP-DEFAULT-NEXT:   }
// CPP-DEFAULT-NEXT:   return [[VAL_6]];
// CPP-DEFAULT-NEXT: }

// CPP-DECLTOP:      int32_t single_use(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]], int32_t [[VAL_4:v[0-9]+]]) {
// CPP-DECLTOP-NEXT:   bool [[VAL_5:v[0-9]+]];
// CPP-DECLTOP-NEXT:   int32_t [[VAL_6:v[0-9]+]];
// CPP-DECLTOP-NEXT:   [[VAL_5]] = bar([[VAL_1]] * M_PI, [[VAL_3]]) - [[VAL_4]] < [[VAL_2]];
// CPP-DECLTOP-NEXT:   ;
// CPP-DECLTOP-NEXT:   if ([[VAL_5]]) {
// CPP-DECLTOP-NEXT:     [[VAL_6]] = [[VAL_1]];
// CPP-DECLTOP-NEXT:   } else {
// CPP-DECLTOP-NEXT:     [[VAL_6]] = [[VAL_1]];
// CPP-DECLTOP-NEXT:   }
// CPP-DECLTOP-NEXT:   return [[VAL_6]];
// CPP-DECLTOP-NEXT: }

func.func @single_use(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> i32 {
  %p0 = emitc.literal "M_PI" : i32
  %e = emitc.expression : i1 {
    %a = emitc.mul %arg0, %p0 : (i32, i32) -> i32
    %b = emitc.call_opaque "bar" (%a, %arg2) : (i32, i32) -> (i32)
    %c = emitc.sub %b, %arg3 : (i32, i32) -> i32
    %d = emitc.cmp lt, %c, %arg1 :(i32, i32) -> i1
    emitc.yield %d : i1
  }
  %v = "emitc.variable"(){value = #emitc.opaque<"">} : () -> i32
  emitc.if %e {
    emitc.assign %arg0 : i32 to %v : i32
    emitc.yield
  } else {
    emitc.assign %arg0 : i32 to %v : i32
    emitc.yield
  }
  return %v : i32
}

// CPP-DEFAULT: int32_t do_not_inline(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]]) {
// CPP-DEFAULT-NEXT: int32_t [[VAL_4:v[0-9]+]] = ([[VAL_1]] + [[VAL_2]]) * [[VAL_3]];
// CPP-DEFAULT-NEXT: return [[VAL_4]];
// CPP-DEFAULT-NEXT:}

// CPP-DECLTOP: int32_t do_not_inline(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]]) {
// CPP-DECLTOP-NEXT: int32_t [[VAL_4:v[0-9]+]];
// CPP-DECLTOP-NEXT: [[VAL_4]] = ([[VAL_1]] + [[VAL_2]]) * [[VAL_3]];
// CPP-DECLTOP-NEXT: return [[VAL_4]];
// CPP-DECLTOP-NEXT:}

func.func @do_not_inline(%arg0: i32, %arg1: i32, %arg2 : i32) -> i32 {
  %e = emitc.expression noinline : i32 {
    %a = emitc.add %arg0, %arg1 : (i32, i32) -> i32
    %b = emitc.mul %a, %arg2 : (i32, i32) -> i32
    emitc.yield %b : i32
  }
  return %e : i32
}

// CPP-DEFAULT:      float paranthesis_for_low_precedence(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]]) {
// CPP-DEFAULT-NEXT:   return (float) ([[VAL_1]] + [[VAL_2]] * [[VAL_3]]);
// CPP-DEFAULT-NEXT: }

// CPP-DECLTOP:      float paranthesis_for_low_precedence(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]]) {
// CPP-DECLTOP-NEXT:   return (float) ([[VAL_1]] + [[VAL_2]] * [[VAL_3]]);
// CPP-DECLTOP-NEXT: }

func.func @paranthesis_for_low_precedence(%arg0: i32, %arg1: i32, %arg2: i32) -> f32 {
  %e = emitc.expression : f32 {
    %a = emitc.add %arg0, %arg1 : (i32, i32) -> i32
    %b = emitc.mul %a, %arg2 : (i32, i32) -> i32
    %d = emitc.cast %b : i32 to f32
    emitc.yield %d : f32
  }
  return %e : f32
}

// CPP-DEFAULT:      int32_t multiple_uses(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]], int32_t [[VAL_4:v[0-9]+]]) {
// CPP-DEFAULT-NEXT:   bool [[VAL_5:v[0-9]+]] = bar([[VAL_1]] * [[VAL_2]], [[VAL_3]]) - [[VAL_4]] < [[VAL_2]];
// CPP-DEFAULT-NEXT:   int32_t [[VAL_6:v[0-9]+]];
// CPP-DEFAULT-NEXT:   if ([[VAL_5]]) {
// CPP-DEFAULT-NEXT:     [[VAL_6]] = [[VAL_1]];
// CPP-DEFAULT-NEXT:   } else {
// CPP-DEFAULT-NEXT:     [[VAL_6]] = [[VAL_1]];
// CPP-DEFAULT-NEXT:   }
// CPP-DEFAULT-NEXT:   bool [[VAL_7:v[0-9]+]];
// CPP-DEFAULT-NEXT:   [[VAL_7]] = [[VAL_5]];
// CPP-DEFAULT-NEXT:   return [[VAL_6]];
// CPP-DEFAULT-NEXT: }

// CPP-DECLTOP:      int32_t multiple_uses(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]], int32_t [[VAL_4:v[0-9]+]]) {
// CPP-DECLTOP-NEXT:   bool [[VAL_5:v[0-9]+]];
// CPP-DECLTOP-NEXT:   int32_t [[VAL_6:v[0-9]+]];
// CPP-DECLTOP-NEXT:   bool [[VAL_7:v[0-9]+]];
// CPP-DECLTOP-NEXT:   [[VAL_5]] = bar([[VAL_1]] * [[VAL_2]], [[VAL_3]]) - [[VAL_4]] < [[VAL_2]];
// CPP-DECLTOP-NEXT:   ;
// CPP-DECLTOP-NEXT:   if ([[VAL_5]]) {
// CPP-DECLTOP-NEXT:     [[VAL_6]] = [[VAL_1]];
// CPP-DECLTOP-NEXT:   } else {
// CPP-DECLTOP-NEXT:     [[VAL_6]] = [[VAL_1]];
// CPP-DECLTOP-NEXT:   }
// CPP-DECLTOP-NEXT:   ;
// CPP-DECLTOP-NEXT:   [[VAL_7]] = [[VAL_5]];
// CPP-DECLTOP-NEXT:   return [[VAL_6]];
// CPP-DECLTOP-NEXT: }

func.func @multiple_uses(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> i32 {
  %e = emitc.expression : i1 {
    %a = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
    %b = emitc.call_opaque "bar" (%a, %arg2) : (i32, i32) -> (i32)
    %c = emitc.sub %b, %arg3 : (i32, i32) -> i32
    %d = emitc.cmp lt, %c, %arg1 :(i32, i32) -> i1
    emitc.yield %d : i1
  }
  %v = "emitc.variable"(){value = #emitc.opaque<"">} : () -> i32
  emitc.if %e {
    emitc.assign %arg0 : i32 to %v : i32
    emitc.yield
  } else {
    emitc.assign %arg0 : i32 to %v : i32
    emitc.yield
  }
  %q = "emitc.variable"(){value = #emitc.opaque<"">} : () -> i1
  emitc.assign %e : i1 to %q : i1
  return %v : i32
}

// CPP-DEFAULT:      int32_t different_expressions(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]], int32_t [[VAL_4:v[0-9]+]]) {
// CPP-DEFAULT-NEXT:   int32_t [[VAL_5:v[0-9]+]] = [[VAL_3]] % [[VAL_4]];
// CPP-DEFAULT-NEXT:   int32_t [[VAL_6:v[0-9]+]] = bar([[VAL_5]], [[VAL_1]] * [[VAL_2]]);
// CPP-DEFAULT-NEXT:   int32_t [[VAL_7:v[0-9]+]];
// CPP-DEFAULT-NEXT:   if ([[VAL_6]] - [[VAL_4]] < [[VAL_2]]) {
// CPP-DEFAULT-NEXT:     [[VAL_7]] = [[VAL_1]];
// CPP-DEFAULT-NEXT:   } else {
// CPP-DEFAULT-NEXT:     [[VAL_7]] = [[VAL_1]];
// CPP-DEFAULT-NEXT:   }
// CPP-DEFAULT-NEXT:   return [[VAL_7]];
// CPP-DEFAULT-NEXT: }

// CPP-DECLTOP:      int32_t different_expressions(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]], int32_t [[VAL_4:v[0-9]+]]) {
// CPP-DECLTOP-NEXT:   int32_t [[VAL_5:v[0-9]+]];
// CPP-DECLTOP-NEXT:   int32_t [[VAL_6:v[0-9]+]];
// CPP-DECLTOP-NEXT:   int32_t [[VAL_7:v[0-9]+]];
// CPP-DECLTOP-NEXT:   [[VAL_5]] = [[VAL_3]] % [[VAL_4]];
// CPP-DECLTOP-NEXT:   [[VAL_6]] = bar([[VAL_5]], [[VAL_1]] * [[VAL_2]]);
// CPP-DECLTOP-NEXT:   ;
// CPP-DECLTOP-NEXT:   if ([[VAL_6]] - [[VAL_4]] < [[VAL_2]]) {
// CPP-DECLTOP-NEXT:     [[VAL_7]] = [[VAL_1]];
// CPP-DECLTOP-NEXT:   } else {
// CPP-DECLTOP-NEXT:     [[VAL_7]] = [[VAL_1]];
// CPP-DECLTOP-NEXT:   }
// CPP-DECLTOP-NEXT:   return [[VAL_7]];
// CPP-DECLTOP-NEXT: }

func.func @different_expressions(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> i32 {
  %e1 = emitc.expression : i32 {
    %a = emitc.rem %arg2, %arg3 : (i32, i32) -> i32
    emitc.yield %a : i32
  }
  %e2 = emitc.expression : i32 {
    %a = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
    %b = emitc.call_opaque "bar" (%e1, %a) : (i32, i32) -> (i32)
    emitc.yield %b : i32
  }
  %e3 = emitc.expression : i1 {
    %c = emitc.sub %e2, %arg3 : (i32, i32) -> i32
    %d = emitc.cmp lt, %c, %arg1 :(i32, i32) -> i1
    emitc.yield %d : i1
  }
  %v = "emitc.variable"(){value = #emitc.opaque<"">} : () -> i32
  emitc.if %e3 {
    emitc.assign %arg0 : i32 to %v : i32
    emitc.yield
  } else {
    emitc.assign %arg0 : i32 to %v : i32
    emitc.yield
  }
  return %v : i32
}

// CPP-DEFAULT:      bool expression_with_address_taken(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t* [[VAL_3]]) {
// CPP-DEFAULT-NEXT:   int32_t [[VAL_4:v[0-9]+]] = [[VAL_1]] % [[VAL_2]];
// CPP-DEFAULT-NEXT:   return &[[VAL_4]] - [[VAL_2]] < [[VAL_3]];
// CPP-DEFAULT-NEXT: }

// CPP-DECLTOP:      bool expression_with_address_taken(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t* [[VAL_3]]) {
// CPP-DECLTOP-NEXT:   int32_t [[VAL_4:v[0-9]+]];
// CPP-DECLTOP-NEXT:   [[VAL_4]] = [[VAL_1]] % [[VAL_2]];
// CPP-DECLTOP-NEXT:   return &[[VAL_4]] - [[VAL_2]] < [[VAL_3]];
// CPP-DECLTOP-NEXT: }

func.func @expression_with_address_taken(%arg0: i32, %arg1: i32, %arg2: !emitc.ptr<i32>) -> i1 {
  %a = emitc.expression : i32 {
    %b = emitc.rem %arg0, %arg1 : (i32, i32) -> i32
    emitc.yield %b : i32
  }
  %c = emitc.expression : i1 {
    %d = emitc.apply "&"(%a) : (i32) -> !emitc.ptr<i32>
    %e = emitc.sub %d, %arg1 : (!emitc.ptr<i32>, i32) -> !emitc.ptr<i32>
    %f = emitc.cmp lt, %e, %arg2 : (!emitc.ptr<i32>, !emitc.ptr<i32>) -> i1
    emitc.yield %f : i1
  }
  return %c : i1
}

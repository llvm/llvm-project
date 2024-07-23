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

// CPP-DEFAULT:      float parentheses_for_low_precedence(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]]) {
// CPP-DEFAULT-NEXT:   return (float) ([[VAL_1]] + [[VAL_2]] * [[VAL_3]]);
// CPP-DEFAULT-NEXT: }

// CPP-DECLTOP:      float parentheses_for_low_precedence(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]]) {
// CPP-DECLTOP-NEXT:   return (float) ([[VAL_1]] + [[VAL_2]] * [[VAL_3]]);
// CPP-DECLTOP-NEXT: }

func.func @parentheses_for_low_precedence(%arg0: i32, %arg1: i32, %arg2: i32) -> f32 {
  %e = emitc.expression : f32 {
    %a = emitc.add %arg0, %arg1 : (i32, i32) -> i32
    %b = emitc.mul %a, %arg2 : (i32, i32) -> i32
    %d = emitc.cast %b : i32 to f32
    emitc.yield %d : f32
  }
  return %e : f32
}

// CPP-DEFAULT:      int32_t parentheses_for_same_precedence(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]]) {
// CPP-DEFAULT-NEXT:   return [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DEFAULT-NEXT: }

// CPP-DECLTOP:      int32_t parentheses_for_same_precedence(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]]) {
// CPP-DECLTOP-NEXT:   return [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DECLTOP-NEXT: }
func.func @parentheses_for_same_precedence(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %e = emitc.expression : i32 {
      %0 = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
      %1 = emitc.div %arg2, %0 : (i32, i32) -> i32
      emitc.yield %1 : i32
    }

  return %e : i32
}

// CPP-DEFAULT:      int32_t user_with_expression_trait(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]]) {
// CPP-DEFAULT-NEXT:   int32_t [[VAL_4:v[0-9]+]] = 0;
// CPP-DEFAULT-NEXT:   int32_t [[EXP_0:v[0-9]+]] = [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DEFAULT-NEXT:   int32_t [[EXP_1:v[0-9]+]] = [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DEFAULT-NEXT:   int32_t [[EXP_2:v[0-9]+]] = [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DEFAULT-NEXT:   int32_t [[EXP_3:v[0-9]+]] = [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DEFAULT-NEXT:   bool [[CAST:v[0-9]+]] = (bool) [[EXP_0]];
// CPP-DEFAULT-NEXT:   int32_t [[ADD:v[0-9]+]] = [[EXP_1]] + [[VAL_4]];
// CPP-DEFAULT-NEXT:   int32_t [[CALL:v[0-9]+]] = bar([[EXP_2]], [[VAL_4]]);
// CPP-DEFAULT-NEXT:   int32_t [[COND:v[0-9]+]] = [[CAST]] ? [[EXP_3]] : [[VAL_4]];
// CPP-DEFAULT-NEXT:   int32_t [[VAR:v[0-9]+]];
// CPP-DEFAULT-NEXT:   [[VAR]] = [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DEFAULT-NEXT:   return [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DEFAULT-NEXT: }

// CPP-DECLTOP:      int32_t user_with_expression_trait(int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]], int32_t [[VAL_3:v[0-9]+]]) {
// CPP-DECLTOP-NEXT:   int32_t [[VAL_4:v[0-9]+]];
// CPP-DECLTOP-NEXT:   int32_t [[EXP_0:v[0-9]+]];
// CPP-DECLTOP-NEXT:   int32_t [[EXP_1:v[0-9]+]];
// CPP-DECLTOP-NEXT:   int32_t [[EXP_2:v[0-9]+]];
// CPP-DECLTOP-NEXT:   int32_t [[EXP_3:v[0-9]+]];
// CPP-DECLTOP-NEXT:   bool [[CAST:v[0-9]+]];
// CPP-DECLTOP-NEXT:   int32_t [[ADD:v[0-9]+]];
// CPP-DECLTOP-NEXT:   int32_t [[CALL:v[0-9]+]];
// CPP-DECLTOP-NEXT:   int32_t [[COND:v[0-9]+]];
// CPP-DECLTOP-NEXT:   int32_t [[VAR:v[0-9]+]];
// CPP-DECLTOP-NEXT:   [[VAL_4]] = 0;
// CPP-DECLTOP-NEXT:   [[EXP_0]] = [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DECLTOP-NEXT:   [[EXP_1]] = [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DECLTOP-NEXT:   [[EXP_2]] = [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DECLTOP-NEXT:   [[EXP_3]] = [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DECLTOP-NEXT:   [[CAST]] = (bool) [[EXP_0]];
// CPP-DECLTOP-NEXT:   [[ADD]] = [[EXP_1]] + [[VAL_4]];
// CPP-DECLTOP-NEXT:   [[CALL]] = bar([[EXP_2]], [[VAL_4]]);
// CPP-DECLTOP-NEXT:   [[COND]] = [[CAST]] ? [[EXP_3]] : [[VAL_4]];
// CPP-DECLTOP-NEXT:   ;
// CPP-DECLTOP-NEXT:   [[VAR]] = [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DECLTOP-NEXT:   return [[VAL_3]] / ([[VAL_1]] * [[VAL_2]]);
// CPP-DECLTOP-NEXT: }
func.func @user_with_expression_trait(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %c0 = "emitc.constant"() {value = 0 : i32} : () -> i32
  %e0 = emitc.expression : i32 {
      %0 = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
      %1 = emitc.div %arg2, %0 : (i32, i32) -> i32
      emitc.yield %1 : i32
    }
  %e1 = emitc.expression : i32 {
      %0 = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
      %1 = emitc.div %arg2, %0 : (i32, i32) -> i32
      emitc.yield %1 : i32
    }
  %e2 = emitc.expression : i32 {
      %0 = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
      %1 = emitc.div %arg2, %0 : (i32, i32) -> i32
      emitc.yield %1 : i32
    }
  %e3 = emitc.expression : i32 {
      %0 = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
      %1 = emitc.div %arg2, %0 : (i32, i32) -> i32
      emitc.yield %1 : i32
    }
  %e4 = emitc.expression : i32 {
      %0 = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
      %1 = emitc.div %arg2, %0 : (i32, i32) -> i32
      emitc.yield %1 : i32
    }
  %e5 = emitc.expression : i32 {
      %0 = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
      %1 = emitc.div %arg2, %0 : (i32, i32) -> i32
      emitc.yield %1 : i32
    }
  %cast = emitc.cast %e0 : i32 to i1
  %add = emitc.add %e1, %c0 : (i32, i32) -> i32
  %call = emitc.call_opaque "bar" (%e2, %c0) : (i32, i32) -> (i32)
  %cond = emitc.conditional %cast, %e3, %c0 : i32
  %var = "emitc.variable"() {value = #emitc.opaque<"">} : () -> i32
  emitc.assign %e4 : i32 to %var : i32
  return %e5 : i32
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

// CPP-DEFAULT: int32_t expression_with_subscript_user(void* [[VAL_1:v.+]])
// CPP-DEFAULT-NEXT:   int64_t [[VAL_2:v.+]] = 0;
// CPP-DEFAULT-NEXT:   int32_t* [[VAL_3:v.+]] = (int32_t*) [[VAL_1]];
// CPP-DEFAULT-NEXT:   return [[VAL_3]][[[VAL_2]]];

func.func @expression_with_subscript_user(%arg0: !emitc.ptr<!emitc.opaque<"void">>) -> i32 {
  %c0 = "emitc.constant"() {value = 0 : i64} : () -> i64
  %0 = emitc.expression : !emitc.ptr<i32> {
    %0 = emitc.cast %arg0 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
    emitc.yield %0 : !emitc.ptr<i32>
  }
  %1 = emitc.subscript %0[%c0] : (!emitc.ptr<i32>, i64) -> i32
  return %1 : i32
}

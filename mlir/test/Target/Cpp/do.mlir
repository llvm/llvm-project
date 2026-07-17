// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT


// CPP-DEFAULT-LABEL: void emitc_do(
// CPP-DEFAULT:         int32_t* [[VAL_1:v[0-9]+]]) {
// CPP-DEFAULT:         int32_t [[VAL_2:v[0-9]+]] = 0;
// CPP-DEFAULT:         do {
// CPP-DEFAULT:           printf("%d", *[[VAL_1]]);
// CPP-DEFAULT:           int32_t [[VAL_3:v[0-9]+]] = [[VAL_2]];
// CPP-DEFAULT:           int32_t [[VAL_4:v[0-9]+]] = [[VAL_3]] + 1;
// CPP-DEFAULT:           [[VAL_2]] = [[VAL_4]];
// CPP-DEFAULT:         } while ([[VAL_2]] <= 10);
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

emitc.func @emitc_do(%arg0 : !emitc.ptr<i32>) {
  %var = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
  %0 = literal "10" : i32
  %1 = literal "1" : i32

  do {
    verbatim "printf(\"%d\", *{});" args %arg0 : !emitc.ptr<i32>
    %var_load = load %var : <i32>
    %tmp_add = add %var_load, %1 : (i32, i32) -> i32
    "emitc.assign"(%var, %tmp_add) : (!emitc.lvalue<i32>, i32) -> ()
  } while {
    %r = expression %var, %0 : (!emitc.lvalue<i32>, i32) -> i1 {
      %var_load = load %var : <i32>
      %cmp = cmp le, %var_load, %0 : (i32, i32) -> i1
      yield %cmp : i1
    }
    
    yield %r : i1
  }

  return
}


// CPP-DEFAULT-LABEL: void emitc_do_with_expression(
// CPP-DEFAULT:         int32_t* [[VAL_1:v[0-9]+]]) {
// CPP-DEFAULT:         int32_t [[VAL_2:v[0-9]+]] = 0;
// CPP-DEFAULT:         int32_t [[VAL_3:v[0-9]+]] = 10 + 1;
// CPP-DEFAULT:         do {
// CPP-DEFAULT:           printf("%d", *[[VAL_1]]);
// CPP-DEFAULT:           int32_t [[VAL_4:v[0-9]+]] = [[VAL_2]];
// CPP-DEFAULT:           int32_t [[VAL_5:v[0-9]+]] = [[VAL_4]] + 1;
// CPP-DEFAULT:           [[VAL_2]] = [[VAL_5]];
// CPP-DEFAULT:         } while ([[VAL_2]] <= [[VAL_3]]);
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

emitc.func @emitc_do_with_expression(%arg0 : !emitc.ptr<i32>) {
  %var = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
  %0 = literal "10" : i32
  %1 = literal "1" : i32

  %add = expression %0, %1 : (i32, i32) -> i32 {
    %add = add %0, %1 : (i32, i32) -> i32
    yield %add : i32
  }

  do {
    verbatim "printf(\"%d\", *{});" args %arg0 : !emitc.ptr<i32>
    %var_load = load %var : <i32>
    %tmp_add = add %var_load, %1 : (i32, i32) -> i32
    "emitc.assign"(%var, %tmp_add) : (!emitc.lvalue<i32>, i32) -> ()
  } while {
    %r = expression %var, %add : (!emitc.lvalue<i32>, i32) -> i1 {
      %var_load = load %var : <i32>
      %cmp = cmp le, %var_load, %add : (i32, i32) -> i1
      yield %cmp : i1
    }

    yield %r : i1
  }

  return
}


// CPP-DEFAULT-LABEL: void emitc_double_do()
// CPP-DEFAULT:         int32_t [[VAL_1:v[0-9]+]] = 0;
// CPP-DEFAULT:         int32_t [[VAL_2:v[0-9]+]] = 0;
// CPP-DEFAULT:         do {
// CPP-DEFAULT:           int32_t [[VAL_3:v[0-9]+]] = [[VAL_1]];
// CPP-DEFAULT:           do {
// CPP-DEFAULT:             int32_t [[VAL_4:v[0-9]+]] = [[VAL_2]];
// CPP-DEFAULT:             printf("i = %d, j = %d", [[VAL_3]], [[VAL_4]]);
// CPP-DEFAULT:             int32_t [[VAL_5:v[0-9]+]] = [[VAL_4]] + 1;
// CPP-DEFAULT:             [[VAL_2]] = [[VAL_5]];
// CPP-DEFAULT:           } while ([[VAL_2]] <= 5);
// CPP-DEFAULT:           int32_t [[VAL_6:v[0-9]+]] = [[VAL_3]] + 1;
// CPP-DEFAULT:           [[VAL_1]] = [[VAL_6]];
// CPP-DEFAULT:         } while ([[VAL_1]] <= 3);
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

emitc.func @emitc_double_do() {
  %var_1 = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
  %var_2 = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
  
  %step = literal "1" : i32
  %end_1 = literal "3" : i32
  %end_2 = literal "5" : i32

  do {
    %var_1_load = load %var_1 : <i32>
    
    do {
      %var_2_load = load %var_2 : <i32>
      verbatim "printf(\"i = %d, j = %d\", {}, {});" args %var_1_load, %var_2_load : i32, i32
      %tmp_add = add %var_2_load, %step : (i32, i32) -> i32
      "emitc.assign"(%var_2, %tmp_add) : (!emitc.lvalue<i32>, i32) -> ()
    } while {
      %r = expression %var_2, %end_2 : (!emitc.lvalue<i32>, i32) -> i1 {
        %var_2_load = load %var_2 : <i32>
        %cmp = cmp le, %var_2_load, %end_2 : (i32, i32) -> i1
        yield %cmp : i1
      }
      
      yield %r : i1
    }

    %tmp_add = add %var_1_load, %step : (i32, i32) -> i32
    "emitc.assign"(%var_1, %tmp_add) : (!emitc.lvalue<i32>, i32) -> ()
  } while {
    %r = expression %var_1, %end_1 : (!emitc.lvalue<i32>, i32) -> i1 {
      %var_1_load = load %var_1 : <i32>
      %cmp = cmp le, %var_1_load, %end_1 : (i32, i32) -> i1
      yield %cmp : i1
    }
    
    yield %r : i1
  }

  return
}


// CPP-DEFAULT-LABEL: bool payload_do_with_empty_body(
// CPP-DEFAULT:         int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]]) {
// CPP-DEFAULT:         bool [[VAL_3:v[0-9]+]] = [[VAL_1]] < [[VAL_2]];
// CPP-DEFAULT:         return [[VAL_3]];
// CPP-DEFAULT:       }
// CPP-DEFAULT:       void emitc_do_with_empty_body(
// CPP-DEFAULT:         int32_t [[VAL_1:v[0-9]+]], int32_t [[VAL_2:v[0-9]+]]) {
// CPP-DEFAULT:         do {
// CPP-DEFAULT:         } while (payload_do_with_empty_body([[VAL_1]], [[VAL_2]]));
// CPP-DEFAULT:         return;
// CPP-DEFAULT:       }

emitc.func @payload_do_with_empty_body(%1 : i32, %2 : i32) -> i1 {
  %cmp = emitc.cmp lt, %1, %2 : (i32, i32) -> i1
  return %cmp : i1
}
func.func @emitc_do_with_empty_body(%arg1 : i32, %arg2 : i32) {
  emitc.do {
  } while {
    %r = emitc.expression %arg1, %arg2 : (i32, i32) -> i1 {
      %call = emitc.call @payload_do_with_empty_body(%arg1, %arg2) : (i32, i32) -> i1
      emitc.yield %call : i1
    }
    emitc.yield %r: i1
  }

  return
}

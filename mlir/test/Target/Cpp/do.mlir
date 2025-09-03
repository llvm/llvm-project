// RUN: mlir-translate -mlir-to-cpp %s | FileCheck --match-full-lines %s -check-prefix=CPP-DEFAULT


// CPP-DEFAULT-LABEL: void emitc_do(int32_t* v1) {
// CPP-DEFAULT:         int32_t v2 = 0;
// CPP-DEFAULT:         do {
// CPP-DEFAULT:           printf("%d", *v1);
// CPP-DEFAULT:           int32_t v3 = v2;
// CPP-DEFAULT:           int32_t v4 = v3 + 1;
// CPP-DEFAULT:           v2 = v4;
// CPP-DEFAULT:         } while (v2 <= 10);
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


// CPP-DEFAULT-LABEL: void emitc_do_with_expression(int32_t* v1) {
// CPP-DEFAULT:         int32_t v2 = 0;
// CPP-DEFAULT:         int32_t v3 = 10 + 1;
// CPP-DEFAULT:         do {
// CPP-DEFAULT:           printf("%d", *v1);
// CPP-DEFAULT:           int32_t v4 = v2;
// CPP-DEFAULT:           int32_t v5 = v4 + 1;
// CPP-DEFAULT:           v2 = v5;
// CPP-DEFAULT:         } while (v2 <= v3);
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


// CPP-DEFAULT-LABEL: void emitc_double_do() {
// CPP-DEFAULT:         int32_t v1 = 0;
// CPP-DEFAULT:         int32_t v2 = 0;
// CPP-DEFAULT:         do {
// CPP-DEFAULT:           int32_t v3 = v1;
// CPP-DEFAULT:           do {
// CPP-DEFAULT:             int32_t v4 = v2;
// CPP-DEFAULT:             printf("i = %d, j = %d", v3, v4);
// CPP-DEFAULT:             int32_t v5 = v4 + 1;
// CPP-DEFAULT:             v2 = v5;
// CPP-DEFAULT:           } while (v2 <= 5);
// CPP-DEFAULT:           int32_t v6 = v3 + 1;
// CPP-DEFAULT:           v1 = v6;
// CPP-DEFAULT:         } while (v1 <= 3);
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

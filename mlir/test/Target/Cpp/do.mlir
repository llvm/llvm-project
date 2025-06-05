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
  %0 = emitc.literal "10" : i32
  %1 = emitc.literal "1" : i32

  emitc.do {
    emitc.verbatim "printf(\"%d\", *{});" args %arg0 : !emitc.ptr<i32>
    %var_load = load %var : <i32>
    %tmp_add = add %var_load, %1 : (i32, i32) -> i32
    "emitc.assign"(%var, %tmp_add) : (!emitc.lvalue<i32>, i32) -> ()
  } while {
    %var_load = load %var : <i32>
    %res = emitc.cmp le, %var_load, %0 : (i32, i32) -> i1
    emitc.yield %res : i1
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
  %0 = emitc.literal "10" : i32
  %1 = emitc.literal "1" : i32

  %add = emitc.expression : i32 {
    %add = add %0, %1 : (i32, i32) -> i32
    yield %add : i32
  }

  emitc.do {
    emitc.verbatim "printf(\"%d\", *{});" args %arg0 : !emitc.ptr<i32>
    %var_load = load %var : <i32>
    %tmp_add = add %var_load, %1 : (i32, i32) -> i32
    "emitc.assign"(%var, %tmp_add) : (!emitc.lvalue<i32>, i32) -> ()
  } while {
    %var_load = load %var : <i32>
    %res = emitc.cmp le, %var_load, %add : (i32, i32) -> i1
    emitc.yield %res : i1
  }

  return
}

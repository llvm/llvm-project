// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

// CHECK-LABEL: test_for_siblings
func.func @test_for_siblings() {
  %start = emitc.literal "0" : index
  %stop = emitc.literal "10" : index
  %step = emitc.literal "1" : index

  %var1 = "emitc.variable"() <{value = 0 : index}> : () -> !emitc.lvalue<index>
  %var2 = "emitc.variable"() <{value = 0 : index}> : () -> !emitc.lvalue<index>

  // CHECK: for (size_t [[ITER0:i[0-9]*]] = {{.*}}; [[ITER0]] < {{.*}}; [[ITER0]] += {{.*}}) {
  emitc.for %i0 = %start to %stop step %step {
    // CHECK: for (size_t [[ITER1:j[0-9]*]] = {{.*}}; [[ITER1]] < {{.*}}; [[ITER1]] += {{.*}}) {
    emitc.for %i1 = %start to %stop step %step {
      // CHECK: {{.*}} = [[ITER0]];
      //"emitc.assign"(%var1,%i0) : (!emitc.lvalue<!emitc.size_t>, !emitc.size_t) -> ()
      emitc.assign %i0 : index to %var1 : !emitc.lvalue<index>
      // CHECK: {{.*}} = [[ITER1]];
      //"emitc.assign"(%var2,%i1) : (!emitc.lvalue<!emitc.size_t>, !emitc.size_t) -> ()
      emitc.assign %i1 : index to %var2 : !emitc.lvalue<index>
    }
  }
  // CHECK: for (size_t [[ITER2:i[0-9]*]] = {{.*}}; [[ITER2]] < {{.*}}; [[ITER2]] += {{.*}})
  emitc.for %ki2 = %start to %stop step %step {
    // CHECK: for (size_t [[ITER3:j[0-9]*]] = {{.*}}; [[ITER3]] < {{.*}}; [[ITER3]] += {{.*}})
    emitc.for %i3 = %start to %stop step %step {
      %1 = emitc.call_opaque "f"() : () -> i32
    }
  }
  return
}

// CHECK-LABEL: test_for_nesting
func.func @test_for_nesting() {
  %start = emitc.literal "0" : index
  %stop = emitc.literal "10" : index
  %step = emitc.literal "1" : index

  // CHECK-COUNT-12: for (size_t [[ITER:[i-t][0-9]*]] = {{.*}}; [[ITER]] < {{.*}}; [[ITER]] += {{.*}}) {
  emitc.for %i0 = %start to %stop step %step {
    emitc.for %i1 = %start to %stop step %step {
      emitc.for %i2 = %start to %stop step %step {
        emitc.for %i3 = %start to %stop step %step {
          emitc.for %i4 = %start to %stop step %step {
            emitc.for %i5 = %start to %stop step %step {
              emitc.for %i6 = %start to %stop step %step {
                emitc.for %i7 = %start to %stop step %step {
                  emitc.for %i8 = %start to %stop step %step {
                    emitc.for %i9 = %start to %stop step %step {
                      emitc.for %i10 = %start to %stop step %step {
                        emitc.for %i11 = %start to %stop step %step {
                          // CHECK: for (size_t [[ITERu0:u13]] = {{.*}}; [[ITERu0]] < {{.*}}; [[ITERu0]] += {{.*}}) {
                          emitc.for %i14 = %start to %stop step %step {
                            // CHECK: for (size_t [[ITERu1:u14]] = {{.*}}; [[ITERu1]] < {{.*}}; [[ITERu1]] += {{.*}}) {
                            emitc.for %i15 = %start to %stop step %step {
                              %0 = emitc.call_opaque "f"() : () -> i32
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return
}

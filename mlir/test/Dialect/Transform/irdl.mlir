// RUN: mlir-opt --transform-interpreter --split-input-file --verify-diagnostics %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.irdl.collect_matching in %arg0 : (!transform.any_op) -> (!transform.any_op){
    ^bb0(%arg1: !transform.any_op):
      irdl.dialect @test {
        irdl.operation @whatever {
          %0 = irdl.is i32
          %1 = irdl.is i64
          %2 = irdl.any_of(%0, %1)
          irdl.results(%2)
        }
      }
    }
    transform.debug.emit_remark_at %0, "matched" : !transform.any_op
    transform.yield
  }

  // expected-remark @below {{matched}}
  "test.whatever"() : () -> i32
  "test.whatever"() : () -> f32
  // expected-remark @below {{matched}}
  "test.whatever"() : () -> i64
}

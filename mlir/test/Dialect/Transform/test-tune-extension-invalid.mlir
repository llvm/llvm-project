// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // expected-error@below {{provided `selected` attribute is not an element of `options` array of attributes}}
    %heads_or_tails = transform.tune.knob<"coin"> = 1 from options = [true, false] -> !transform.any_param
    transform.yield
  }
}

// -----

func.func private @f()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // expected-error@below {{non-deterministic choice "coin" is only resolved through providing a `selected` attr}}
    %heads_or_tails = transform.tune.knob<"coin"> options = [true, false] -> !transform.any_param
    transform.yield
  }
}

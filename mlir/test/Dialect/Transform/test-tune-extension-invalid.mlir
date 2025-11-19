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

// -----

func.func private @f()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // expected-error@below {{'selected_region' attribute specifies region at index 2 while op has only 2 regions}}
    transform.tune.alternatives<"bifurcation"> selected_region = 2 {
      transform.yield
    }, {
      transform.yield
    }
    transform.yield
  }
}

// -----

func.func private @f()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %singleton_of_c0 = transform.param.constant [0] -> !transform.any_param
    // expected-error@below {{param should hold exactly one integer attribute, got: [0]}}
    transform.tune.alternatives<"bifurcation"> selected_region = %singleton_of_c0 : !transform.any_param {
      transform.yield
    }, {
      transform.yield
    }
    transform.yield
  }
}

// -----

func.func private @f()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %c0 = transform.param.constant 0 -> !transform.any_param
    %c1 = transform.param.constant 1 -> !transform.any_param
    %c0_and_c1 = transform.merge_handles %c0, %c1 : !transform.any_param
    // expected-error@below {{param should hold exactly one integer attribute}}
    transform.tune.alternatives<"bifurcation"> selected_region = %c0_and_c1 : !transform.any_param {
      transform.yield
    }, {
      transform.yield
    }
    transform.yield
  }
}

// -----

func.func private @f()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %c2 = transform.param.constant 2 -> !transform.any_param
    // expected-error@below {{'selected_region' attribute/param specifies region at index 2 while op has only 2 regions}}
    transform.tune.alternatives<"bifurcation"> selected_region = %c2 : !transform.any_param {
      transform.yield
    }, {
      transform.yield
    }
    transform.yield
  }
}

// -----

func.func private @f()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // expected-error@below {{non-deterministic choice "bifurcation" is only resolved through providing a `selected_region` attr/param}}
    transform.tune.alternatives<"bifurcation"> {
      transform.yield
    }, {
      transform.yield
    }
    transform.yield
  }
}

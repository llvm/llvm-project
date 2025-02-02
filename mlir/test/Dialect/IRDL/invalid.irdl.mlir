// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// Testing invalid IRDL IRs

irdl.dialect @testd {
  irdl.type @type {
    %0 = irdl.any
    // expected-error@+1 {{expected valid keyword}}
    irdl.parameters(%0)
  }
}

// -----

irdl.dialect @testd {
  irdl.type @type {
    %0 = irdl.any
    // expected-error@+1 {{expected valid keyword}}
    irdl.parameters(123: %0)
  }
}

// -----

irdl.dialect @testd {
  irdl.type @type {
    %0 = irdl.any
    // expected-error@+1 {{name of parameter #0 must contain only letters, digits and underscores}}
    irdl.parameters(test$test: %0)
  }
}

// -----

irdl.dialect @testd {
  irdl.operation @op {
    %0 = irdl.any
    // expected-error@+1 {{name of result #0 must contain only letters, digits and underscores}}
    irdl.results(test$test: %0)
  }
}

// -----

irdl.dialect @testd {
  irdl.operation @op {
    %0 = irdl.any
    // expected-error@+1 {{name of operand #0 must contain only letters, digits and underscores}}
    irdl.operands(test$test: %0)
  }
}

// -----

irdl.dialect @testd {
  irdl.type @type {
    %0 = irdl.any
    // expected-error@+1 {{name of parameter #2 is a duplicate of the name of parameter #0}}
    irdl.parameters(foo: %0, bar: %0, foo: %0)
  }
}

// -----

irdl.dialect @testd {
  irdl.operation @op {
    %0 = irdl.any
    // expected-error@+1 {{name of result #2 is a duplicate of the name of result #0}}
    irdl.results(foo: %0, bar: %0, foo: %0)
  }
}

// -----

irdl.dialect @testd {
  irdl.operation @op {
    %0 = irdl.any
    // expected-error@+1 {{name of operand #2 is a duplicate of the name of operand #0}}
    irdl.operands(foo: %0, bar: %0, foo: %0)
  }
}

// -----

irdl.dialect @testd {
  // expected-error@+1 {{contains a value named 'foo' for both its operands and results}}
  irdl.operation @op {
    %0 = irdl.any
    irdl.operands(foo: %0)
    irdl.results(foo: %0)
  }
}

// -----

irdl.dialect @testd {
  // expected-error@+1 {{contains a value named 'bar' for both its regions and results}}
  irdl.operation @op {
    %0 = irdl.any
    %1 = irdl.region
    irdl.regions(bar: %1)
    irdl.results(bar: %0)
  }
}

// -----

irdl.dialect @testd {
  // expected-error@+1 {{contains a value named 'baz' for both its regions and operands}}
  irdl.operation @op {
    %0 = irdl.any
    %1 = irdl.region
    irdl.regions(baz: %1)
    irdl.operands(baz: %0)
  }
}

// -----

irdl.dialect @testd {
  // expected-error@+1 {{contains a value named 'baz' for both its regions and results}}
  irdl.operation @op {
    %0 = irdl.any
    %1 = irdl.region
    irdl.regions(baz: %1)
    irdl.operands(qux: %0)
    irdl.results(baz: %0)
  }
}

// -----

irdl.dialect @testd {
  irdl.type @type {
    // expected-error@+1 {{symbol '@foo' not found}}
    %0 = irdl.base @foo
    irdl.parameters(foo: %0)
  }
}

// -----

irdl.dialect @testd {
  irdl.type @type {
    // expected-error@+1 {{the base type or attribute name should start with '!' or '#'}}
    %0 = irdl.base "builtin.integer"
    irdl.parameters(foo: %0)
  }
}

// -----

irdl.dialect @testd {
  irdl.type @type {
    // expected-error@+1 {{the base type or attribute name should start with '!' or '#'}}
    %0 = irdl.base ""
    irdl.parameters(foo: %0)
  }
}

// -----

irdl.dialect @testd {
  irdl.type @type {
    // expected-error@+1 {{the base type or attribute should be specified by either a name}}
    %0 = irdl.base
    irdl.parameters(foo: %0)
  }
}

// -----

func.func private @not_a_type_or_attr()

irdl.dialect @invalid_parametric {
  irdl.operation @foo {
    // expected-error@+1 {{symbol '@not_a_type_or_attr' does not refer to a type or attribute definition}}
    %param = irdl.parametric @not_a_type_or_attr<>
    irdl.results(foo: %param)
  }
}

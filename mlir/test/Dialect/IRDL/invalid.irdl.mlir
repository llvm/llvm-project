// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// Testing invalid IRDL IRs

func.func private @foo()

irdl.dialect @testd {
  irdl.type @type {
    // expected-error@+1 {{symbol '@foo' not found}}
    %0 = irdl.base @foo
    irdl.parameters(%0)
  }
}

// -----

irdl.dialect @testd {
  irdl.type @type {
    // expected-error@+1 {{the base type or attribute name should start with '!' or '#'}}
    %0 = irdl.base "builtin.integer"
    irdl.parameters(%0)
  }
}

// -----

irdl.dialect @testd {
  irdl.type @type {
    // expected-error@+1 {{the base type or attribute name should start with '!' or '#'}}
    %0 = irdl.base ""
    irdl.parameters(%0)
  }
}

// -----

irdl.dialect @testd {
  irdl.type @type {
    // expected-error@+1 {{the base type or attribute should be specified by either a name}}
    %0 = irdl.base
    irdl.parameters(%0)
  }
}

// -----

irdl.dialect @invalid_parametric {
  irdl.operation @foo {
    // expected-error@+1 {{symbol '@not_a_type_or_attr' does not refer to a type or attribute definition}}
    %param = irdl.parametric @not_a_type_or_attr<>
    irdl.results(%param)
  }

  irdl.operation @not_a_type_or_attr {
    %param = irdl.is i1
    irdl.results(%param)
  }
}

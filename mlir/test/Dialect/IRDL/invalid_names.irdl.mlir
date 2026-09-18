// RUN: mlir-irdl-to-cpp %s --verify-diagnostics --split-input-file
// expected-error@+1 {{name of dialect must start with a lowercase letter}}
irdl.dialect @_no_leading_underscore {
}

// -----

// expected-error@+1 {{name of dialect must start with a lowercase letter}}
irdl.dialect @NoUpperCase {
}

// -----

irdl.dialect @test_dialect {
  // expected-error@+1 {{name of operation must start with a lowercase letter}}
  irdl.operation @_no_leading_underscore {
    %0 = irdl.any
    irdl.results(res: %0)
  }
}

// -----

irdl.dialect @test_dialect {
  // expected-error@+1 {{name of operation must contain only lowercase letters, digits, underscores, dollar signs or dots}}
  irdl.operation @noUpperCase {}
}

// -----

irdl.dialect @test_dialect {
    // expected-error@+1 {{name of operation must start with a lowercase letter}}
    irdl.operation @NoUpperCase {
        %0 = irdl.any
        irdl.results(res: %0)
    }
}

// -----

irdl.dialect @test_dialect {
  irdl.operation @test_op {
    %0 = irdl.any
    // expected-error@+1 {{name of result #0 must start with a lowercase letter}}
    irdl.results(_no_leading_underscore: %0)
  }
}

// -----

irdl.dialect @test_dialect {
  irdl.operation @test_op {
    %0 = irdl.any
    // expected-error@+1 {{name of result #0 must start with a lowercase letter}}
    irdl.results(NoUpperCase: %0)
  }
}

// -----

irdl.dialect @test_dialect {
  // expected-error@+1 {{name of type is empty}}
  irdl.type @"" {
    %0 = irdl.any
  }
}

// -----

irdl.dialect @test_dialect {
  // expected-error@+1 {{name of attribute is empty}}
  irdl.attribute @"" {
    %0 = irdl.any
  }
}

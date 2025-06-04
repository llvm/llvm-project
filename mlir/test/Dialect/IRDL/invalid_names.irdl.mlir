// RUN: mlir-irdl-to-cpp %s --verify-diagnostics --split-input-file
// expected-error@+1 {{name of dialect should not contain leading or double underscores}}
irdl.dialect @_no_leading_underscore {
}

// -----

// expected-error@+1 {{name of dialect should not contain leading or double underscores}}
irdl.dialect @no__double__underscores {
}

// -----

// expected-error@+1 {{name of dialect should not contain uppercase letters}}
irdl.dialect @NoUpperCase {
}

// -----

// expected-error@+1 {{name of dialect must contain only lowercase letters, digits and underscores}}
irdl.dialect @no_weird_symbol$ {
}

// -----

irdl.dialect @test_dialect {
  // expected-error@+1 {{name of operation should not contain leading or double underscores}}
  irdl.operation @_no_leading_underscore {
    %0 = irdl.any
    irdl.results(res: %0)
  }
}

// -----

irdl.dialect @test_dialect {
  // expected-error@+1 {{name of operation should not contain leading or double underscores}}
  irdl.operation @no__double__underscores {
    %0 = irdl.any
    irdl.results(res: %0)
  }
}

// -----

irdl.dialect @test_dialect {
    // expected-error@+1 {{name of operation should not contain uppercase letters}}
    irdl.operation @NoUpperCase {
        %0 = irdl.any
        irdl.results(res: %0)
    }
}

// -----

irdl.dialect @test_dialect {
  // expected-error@+1 {{name of operation must contain only lowercase letters, digits and underscores}}
  irdl.operation @no_weird_symbol$ {
    %0 = irdl.any
    irdl.results(res: %0)
  }
}

// -----

irdl.dialect @test_dialect {
  irdl.operation @test_op {
    %0 = irdl.any
    // expected-error@+1 {{name of result #0 should not contain leading or double underscores}}
    irdl.results(_no_leading_underscore: %0)
  }
}

// -----

irdl.dialect @test_dialect {
  irdl.operation @test_op {
    %0 = irdl.any
    // expected-error@+1 {{name of result #0 should not contain leading or double underscores}}
    irdl.results(no__double__underscores: %0)
  }
}

// -----

irdl.dialect @test_dialect {
  irdl.operation @test_op {
    %0 = irdl.any
    // expected-error@+1 {{name of result #0 should not contain uppercase letters}}
    irdl.results(NoUpperCase: %0)
  }
}

// -----

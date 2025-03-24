// RUN: mlir-irdl-to-cpp %s --verify-diagnostics --split-input-file
// expected-error@+1 {{name of dialect "_no_leading_underscore" should not contain leading or double underscores}}
irdl.dialect @_no_leading_underscore {
}

// -----

// expected-error@+1 {{name of dialect "no__double__underscores" should not contain leading or double underscores}}
irdl.dialect @no__double__underscores {
}

// -----

// expected-error@+1 {{name of dialect "NoUpperCase" should not contain uppercase letters}}
irdl.dialect @NoUpperCase {
}

// -----

// expected-error@+1 {{name of dialect "no_weird_symbol$" must contain only lowercase letters, digits and underscores}}
irdl.dialect @no_weird_symbol$ {
}

// -----

irdl.dialect @test_dialect {
    // expected-error@+1 {{name of operation "_no_leading_underscore" should not contain leading or double underscores}}
    irdl.operation @_no_leading_underscore {
        %0 = irdl.any
        irdl.results(res: %0)
    }
}

// -----

irdl.dialect @test_dialect {
    // expected-error@+1 {{name of operation "no__double__underscores" should not contain leading or double underscores}}
    irdl.operation @no__double__underscores {
        %0 = irdl.any
        irdl.results(res: %0)
    }
}

// -----

irdl.dialect @test_dialect3 {
    // expected-error@+1 {{name of operation "NoUpperCase" should not contain uppercase letters}}
    irdl.operation @NoUpperCase {
        %0 = irdl.any
        irdl.results(res: %0)
    }
}

// -----

irdl.dialect @test_dialect4 {
    // expected-error@+1 {{name of operation "no_weird_symbol$" must contain only lowercase letters, digits and underscores}}
    irdl.operation @no_weird_symbol$ {
        %0 = irdl.any
        irdl.results(res: %0)
    }
}

// -----

irdl.dialect @test_dialect5 {
    irdl.operation @test_op {
        %0 = irdl.any
        // expected-error@+1 {{name of result "_no_leading_underscore" should not contain leading or double underscores}}
        irdl.results(_no_leading_underscore: %0)
    }
}

// -----

irdl.dialect @test_dialect6 {
    irdl.operation @test_op {
        %0 = irdl.any
        // expected-error@+1 {{name of result "no__double__underscores" should not contain leading or double underscores}}
        irdl.results(no__double__underscores: %0)
    }
}

// -----

irdl.dialect @test_dialect7 {
    irdl.operation @test_op {
        %0 = irdl.any
        // expected-error@+1 {{name of result "NoUpperCase" should not contain uppercase letters}}
        irdl.results(NoUpperCase: %0)
    }
}

// -----

// RUN: mlir-irdl-to-cpp %s --verify-diagnostics
// expected-error@+1 {{Error in symbol name `_no_leading_underscore`: No leading or double underscores allowed.}}
irdl.dialect @_no_leading_underscore {
}

// -----

// expected-error@+1 {{Error in symbol name `_no__double__underscores`: No leading or double underscores allowed.}}
irdl.dialect @_no__double__underscores {
}

// -----

// expected-error@+1 {{Error in symbol name `NoUpperCase`: Upper-case characters are not allowed}}
irdl.dialect @NoUpperCase {
}

// -----

// expected-error@+1 {{Error in symbol name `no_weird_symbol$`: Only numbers and lower-case characters allowed}}
irdl.dialect @no_weird_symbol$ {
}

// -----

irdl.dialect @test_dialect {
    // expected-error@+1 {{Error in symbol name `_no_leading_underscore`: No leading or double underscores allowed.}}
    irdl.operation @_no_leading_underscore {
        %0 = irdl.any
        irdl.results(res: %0)
    }
}

// -----

irdl.dialect @test_dialect2 {
    // expected-error@+1 {{Error in symbol name `_no__double__underscores`: No leading or double underscores allowed.}}
    irdl.operation @_no__double__underscores {
        %0 = irdl.any
        irdl.results(res: %0)
    }
}

// -----

irdl.dialect @test_dialect3 {
    // expected-error@+1 {{Error in symbol name `NoUpperCase`: Upper-case characters are not allowed.}}
    irdl.operation @NoUpperCase {
        %0 = irdl.any
        irdl.results(res: %0)
    }
}

// -----

irdl.dialect @test_dialect4 {
    // expected-error@+1 {{Error in symbol name `no_weird_symbol$`: Only numbers and lower-case characters allowed.}}
    irdl.operation @no_weird_symbol$ {
        %0 = irdl.any
        irdl.results(res: %0)
    }
}

// -----

irdl.dialect @test_dialect5 {
    irdl.operation @test_op {
        %0 = irdl.any
        // expected-error@+1 {{Error in symbol name `_no_leading_underscore`: No leading or double underscores allowed.}}
        irdl.results(_no_leading_underscore: %0)
    }
}

// -----

irdl.dialect @test_dialect6 {
    irdl.operation @test_op {
        %0 = irdl.any
        // expected-error@+1 {{Error in symbol name `_no__double__underscores`: No leading or double underscores allowed.}}
        irdl.results(_no__double__underscores: %0)
    }
}

// -----

irdl.dialect @test_dialect7 {
    irdl.operation @test_op {
        %0 = irdl.any
        // expected-error@+1 {{Error in symbol name `NoUpperCase`: Upper-case characters are not allowed.}}
        irdl.results(NoUpperCase: %0)
    }
}

// -----

// RUN: mlir-irdl-to-cpp %s --verify-diagnostics
irdl.dialect @test_irdl_to_cpp {
    irdl.operation @results_no_any_of {
        %0 = irdl.any
        // expected-error@+1 {{IRDL C++ translation only supports irdl.any constraint for types}}
        %1 = irdl.any_of(%0, %0)
        irdl.results(res: %1)
    }
}
// ----- 

irdl.dialect @test_irdl_to_cpp_2 {
    irdl.operation @operands_no_any_of {
        %0 = irdl.any
        // expected-error@+1 {{IRDL C++ translation only supports irdl.any constraint for types}}
        %1 = irdl.all_of(%0, %0)
        irdl.operands(test: %1)
        irdl.results(res: %0)
    }
}

// -----

irdl.dialect @test_irdl_to_cpp_3 {
    // expected-error@+1 {{IRDL C++ translation does not yet support attributes.}}
    irdl.attribute @no_attrs
}

// -----

irdl.dialect @test_irdl_to_cpp_4 {
    irdl.operation @test_op {
        %0 = irdl.any
        // expected-error@+1 {{IRDL C++ translation does not yet support attributes.}}
        irdl.attributes {
            "attr" = %0
        }
    }
}

// -----

irdl.dialect @test_irdl_to_cpp_5 {
    irdl.type @ty {
        %0 = irdl.any 
        // expected-error@+1 {{IRDL C++ translation does not yet support type parameters.}}
        irdl.parameters(ty: %0)
    }
}

// -----

irdl.dialect @test_irdl_to_cpp_6 {
    irdl.operation @test_op {
        // expected-error@+1 {{IRDL C++ translation does not yet support regions.}}
        %0 = irdl.region()
        irdl.regions(reg: %0)
    }
    
}

// -----

irdl.dialect @test_irdl_to_cpp_7 {
    irdl.operation @test_op {
        // expected-error@+1 {{IRDL C++ translation does not yet support regions.}}
        irdl.regions()
    }
    
}

// -----

irdl.dialect @test_irdl_to_cpp_8 {
    irdl.type @test_derived {
        // expected-error@+1 {{IRDL C++ translation does not yet support base types.}}
        %0 = irdl.base "!builtin.integer"
    }
    
}


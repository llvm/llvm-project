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

// no support for split-buffer yet
irdl.dialect @test_irdl_to_cpp_2 {
    irdl.operation @operands_no_any_of {
        %0 = irdl.any
        // expected-error@+1 {{IRDL C++ translation only supports irdl.any constraint for types}}
        %1 = irdl.any_of(%0, %0)
        irdl.operands(test: %1)
        irdl.results(res: %0)
    }
}

// -----

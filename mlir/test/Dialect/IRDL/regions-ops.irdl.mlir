// RUN: mlir-opt %s -verify-diagnostics -split-input-file

irdl.dialect @testRegionOpNegativeNumber {
    irdl.operation @op {
        // expected-error @below {{'irdl.region' op the number of blocks is expected to be >= 1 but got -42}}
        %r1 = irdl.region with size -42
    }
}

// -----

irdl.dialect @testRegionsOpMissingName {
    irdl.operation @op {
        %r1 = irdl.region
        // expected-error @below {{expected valid keyword}}
        irdl.regions(%r1)
    }
}

// -----

irdl.dialect @testRegionsOpWrongName {
    irdl.operation @op {
        %r1 = irdl.region
        // expected-error @below {{name of region #0 must contain only letters, digits and underscores}}
        irdl.regions(test$test: %r1)
    }
}

// -----

irdl.dialect @testRegionsDuplicateName {
    irdl.operation @op {
        %r1 = irdl.region
        // expected-error @below {{name of region #2 is a duplicate of the name of region #0}}
        irdl.regions(foo: %r1, bar: %r1, foo: %r1)
    }
}

// -----

irdl.dialect @testRegionsOpWrongOperation {
    irdl.operation @op {
        // expected-note @below {{prior use here}}
        %r1 = irdl.any
        // expected-error @below {{use of value '%r1' expects different type than prior uses: '!irdl.region' vs '!irdl.attribute'}}
        irdl.regions(foo: %r1)
    }
}

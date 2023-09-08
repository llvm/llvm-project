// RUN: mlir-opt %s -verify-diagnostics -split-input-file

irdl.dialect @testRegionOpNegativeNumber {
    irdl.operation @op {
        // expected-error @below {{'irdl.region' op the number of blocks is expected to be >= 1 but got -42}}
        %r1 = irdl.region with size -42
    }
}

// -----

irdl.dialect @testRegionsOpWrongOperation {
    irdl.operation @op {
        // expected-note @below {{prior use here}}
        %r1 = irdl.any
        // expected-error @below {{use of value '%r1' expects different type than prior uses: '!irdl.region' vs '!irdl.attribute'}}
        irdl.regions(%r1)
    }
}

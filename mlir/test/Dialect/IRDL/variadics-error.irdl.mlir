// RUN: mlir-opt %s -verify-diagnostics -split-input-file

irdl.dialect @errors {
  irdl.operation @operands {
    %0 = irdl.is i32

    // expected-error@+1 {{'irdl.operands' op the number of operands and their variadicities must be the same, but got 2 and 1 respectively}}
    "irdl.operands"(%0, %0) <{names = ["foo", "bar"], variadicity = #irdl<variadicity_array[single]>}> : (!irdl.attribute, !irdl.attribute) -> ()
  }
}

// -----

irdl.dialect @errors {
  irdl.operation @operands2 {
    %0 = irdl.is i32
   
    // expected-error@+1 {{'irdl.operands' op the number of operands and their variadicities must be the same, but got 1 and 2 respectively}}
    "irdl.operands"(%0) <{names = ["foo"], variadicity = #irdl<variadicity_array[single, single]>}> : (!irdl.attribute) -> ()
  }
}

// -----

irdl.dialect @errors {
  irdl.operation @results {
    %0 = irdl.is i32

    // expected-error@+1 {{'irdl.results' op the number of results and their variadicities must be the same, but got 2 and 1 respectively}}
    "irdl.results"(%0, %0) <{names = ["foo", "bar"], variadicity = #irdl<variadicity_array[single]>}> : (!irdl.attribute, !irdl.attribute) -> ()
  }
}

// -----

irdl.dialect @errors {
  irdl.operation @results2 {
    %0 = irdl.is i32
   
    // expected-error@+1 {{'irdl.results' op the number of results and their variadicities must be the same, but got 1 and 2 respectively}}
    "irdl.results"(%0) <{names = ["foo"], variadicity = #irdl<variadicity_array[single, single]>}> : (!irdl.attribute) -> ()
  }
}

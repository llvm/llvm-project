// RUN: mlir-opt %s -verify-diagnostics -split-input-file

irdl.dialect @errors {
  irdl.operation @operands {
    %0 = irdl.is i32

    // expected-error@+1 {{'irdl.operands' op the number of operands and their variadicities must be the same, but got 2 and 1 respectively}}
    "irdl.operands"(%0, %0) <{variadicity = #irdl<variadicity_array[single]>}> : (!irdl.attribute, !irdl.attribute) -> ()
  }
}

// -----

irdl.dialect @errors {
  irdl.operation @operands2 {
    %0 = irdl.is i32

    // expected-error@+1 {{'irdl.operands' op the number of operands and their variadicities must be the same, but got 1 and 2 respectively}}
    "irdl.operands"(%0) <{variadicity = #irdl<variadicity_array[single, single]>}> : (!irdl.attribute) -> ()
  }
}

// -----

irdl.dialect @errors {
  irdl.operation @results {
    %0 = irdl.is i32

    // expected-error@+1 {{'irdl.results' op the number of operands and their variadicities must be the same, but got 2 and 1 respectively}}
    "irdl.results"(%0, %0) <{variadicity = #irdl<variadicity_array[single]>}> : (!irdl.attribute, !irdl.attribute) -> ()
  }
}

// -----

irdl.dialect @errors {
  irdl.operation @results2 {
    %0 = irdl.is i32

    // expected-error@+1 {{'irdl.results' op the number of operands and their variadicities must be the same, but got 1 and 2 respectively}}
    "irdl.results"(%0) <{variadicity = #irdl<variadicity_array[single, single]>}> : (!irdl.attribute) -> ()
  }
}

// -----

irdl.dialect @errors {
  irdl.operation @no_var {
    %0 = irdl.is i32
    %1 = irdl.is i64

    // expected-error@+1 {{'irdl.attributes' op requires attribute 'variadicity'}}
    "irdl.attributes"(%0) <{attributeValueNames = ["attr1"]}> : (!irdl.attribute) -> ()
  }
}

// -----

irdl.dialect @errors {
  irdl.operation @attrs1 {
    %0 = irdl.is i32
    %1 = irdl.is i64

    // expected-error@+1 {{'irdl.attributes' op the number of attributes and their variadicities must be the same, but got 2 and 1 respectively}}
    "irdl.attributes"(%0, %1) <{attributeValueNames = ["attr1", "attr2"], variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute, !irdl.attribute) -> ()
  }
}

// -----

irdl.dialect @errors {
  irdl.operation @attrs2 {
    %0 = irdl.is i32
    %1 = irdl.is i64

    // expected-error@+1 {{'irdl.attributes' op the number of attributes and their variadicities must be the same, but got 1 and 2 respectively}}
    "irdl.attributes"(%0) <{attributeValueNames = ["attr1"], variadicity = #irdl<variadicity_array[ single, single]>}> : (!irdl.attribute) -> ()
  }
}


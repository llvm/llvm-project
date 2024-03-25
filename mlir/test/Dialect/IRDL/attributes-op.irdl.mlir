// RUN: mlir-opt %s -verify-diagnostics -split-input-file
irdl.dialect @errors {
  irdl.operation @attrs1 {
    %0 = irdl.is i32
    %1 = irdl.is i64

    // expected-error@+1 {{'irdl.attributes' op the number of attribute names and their constraints must be the same but got 1 and 2 respectively}}
    "irdl.attributes"(%0, %1) <{attributeValueNames = ["attr1"]}> : (!irdl.attribute, !irdl.attribute) -> ()
  }
}

// -----

irdl.dialect @errors {
  irdl.operation @attrs2 {
    %0 = irdl.is i32
    %1 = irdl.is i64
   
    // expected-error@+1 {{'irdl.attributes' op the number of attribute names and their constraints must be the same but got 2 and 1 respectively}}
    "irdl.attributes"(%0) <{attributeValueNames = ["attr1", "attr2"]}> : (!irdl.attribute) -> ()
  }
}

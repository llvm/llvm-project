// RUN: mlir-opt -split-input-file --mlgo-add-reflection-map="field-attr-name=emitc.field_ref \
// RUN: excluded-field-attrs="emitc.other_field"" -verify-diagnostics %s | FileCheck %s

/// Tests that a reflection map is created for fields with a certain attribute.

emitc.class @actionClass {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.field_ref = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.field_ref = ["some_feature"]}
  emitc.func @"operator()"() {
    %0 = get_field @fieldName0 : !emitc.array<1xf32>
    return
  }
}

// CHECK:       emitc.class @actionClass {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
// CHECK-NEXT:    emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.field_ref = ["some_feature"]}
// CHECK-NEXT:    emitc.field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>"> = 
// CHECK-SAME:    #emitc.opaque<"{ { \22another_feature\22, reinterpret_cast<char*>(&fieldName0) }, { \22some_feature\22, reinterpret_cast<char*>(&fieldName1) } }">  
// CHECK-NEXT:    emitc.func @getBufferForName(%{{.*}}: !emitc.opaque<"std::string">) -> !emitc.ptr<!emitc.opaque<"char">> {
// CHECK-NEXT:      %[[MAP0:.*]] = get_field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>">
// CHECK-NEXT:      %[[VAL0:.*]] = member_call_opaque %[[MAP0]] "at"({{.*}}) : !emitc.opaque<"const std::map<std::string, char*>">, (!emitc.opaque<"std::string">) -> !emitc.ptr<!emitc.opaque<"char">>
// CHECK-NEXT:      return %[[VAL0]] : !emitc.ptr<!emitc.opaque<"char">>
// CHECK-NEXT:    }
// CHECK-NEXT:    emitc.func @"operator()"() {
// CHECK-NEXT:      %{{.*}} = get_field @fieldName0 : !emitc.array<1xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Test that a reflection map is created for fields with a certain named attribute
/// but not ones with an attribute present in the ignore-attributes option.

emitc.class @actionClassExcluded {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.field_ref = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.other_field = ["some_feature"]}
  emitc.func @"operator()"() {
    %0 = get_field @fieldName0 : !emitc.array<1xf32>
    return
  }
}

// CHECK:       emitc.class @actionClassExcluded {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
// CHECK-NEXT:    emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.other_field = ["some_feature"]}
// CHECK-NEXT:    emitc.field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>"> = 
// CHECK-SAME:    #emitc.opaque<"{ { \22another_feature\22, reinterpret_cast<char*>(&fieldName0) } }">  
// CHECK-NEXT:    emitc.func @getBufferForName(%{{.*}}: !emitc.opaque<"std::string">) -> !emitc.ptr<!emitc.opaque<"char">> {
// CHECK-NEXT:      %[[MAP1:.*]] = get_field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>">
// CHECK-NEXT:      %[[VAL1:.*]] = member_call_opaque %[[MAP1]] "at"({{.*}}) : !emitc.opaque<"const std::map<std::string, char*>">, (!emitc.opaque<"std::string">) -> !emitc.ptr<!emitc.opaque<"char">>
// CHECK-NEXT:      return %[[VAL1]] : !emitc.ptr<!emitc.opaque<"char">>
// CHECK-NEXT:    }
// CHECK-NEXT:    emitc.func @"operator()"() {
// CHECK-NEXT:      %{{.*}} = get_field @fieldName0 : !emitc.array<1xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Test that the pass leaves IR unchanged if fields don't have any attributes

emitc.class @actionClassNoAttrs {
  // expected-error @below {{FieldOp must have a dictionary attribute named 'emitc.field_ref' with an array containing a string attribute}}
  emitc.field @fieldName0 : !emitc.array<1xf32>
  emitc.func @"operator()"() {
    return
  }
}

// CHECK-LABEL: emitc.class @actionClassNoAttrs {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32>
// CHECK-NEXT:    emitc.func @"operator()"() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Test that the pass leaves IR unchanged if the ClassOp doesn't have any fields

emitc.class @actionClassNoFields {
  emitc.func @"operator()"() {
    return
  }
}

// CHECK-LABEL: emitc.class @actionClassNoFields {
// CHECK-NEXT:    emitc.func @"operator()"() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Test that the pass returns with a match failure if the ClassOp doesn't have
/// a FunctionOp named operator()

// expected-error @below {{ClassOp must contain a function named 'operator()' to add reflection map}}
emitc.class @actionClassNoOperator {
  emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
}

// CHECK-LABEL: emitc.class @actionClassNoOperator {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
// CHECK-NEXT:  }

// -----

/// Test that the pass returns with a match failure if a FieldOp has the specified
/// dictionary attribute with an array containing a type other than string

emitc.class @actionClassNonStringAttr {
  // expected-error @below {{FieldOp must have a dictionary attribute named 'emitc.field_ref' with an array containing a string attribute}}
  emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = [1]}
  emitc.func @"operator()"() {
    return
  }
}

// CHECK-LABEL: emitc.class @actionClassNonStringAttr {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = [1]}
// CHECK-NEXT:    emitc.func @"operator()"() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
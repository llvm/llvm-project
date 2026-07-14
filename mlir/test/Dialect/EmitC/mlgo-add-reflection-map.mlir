// RUN: mlir-opt -split-input-file --mlgo-add-reflection-map="field-attr-name=emitc.field_ref \
// RUN: excluded-field-attrs="emitc.other_field"" %s | FileCheck %s '-D$QUOTE=\22'

/// Tests that a reflection map is created for fields with a certain attribute.

emitc.class @foo {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.field_ref = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.field_ref = ["some_feature"]}
  emitc.func @"operator()"() {
    %0 = get_field @fieldName0 : !emitc.array<1xf32>
    return
  }
}

// CHECK:       emitc.class @foo {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
// CHECK-NEXT:    emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.field_ref = ["some_feature"]}
// CHECK-NEXT:    emitc.field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>"> = 
// CHECK-SAME:    #emitc.opaque<"{ { [[$QUOTE]]another_feature[[$QUOTE]], reinterpret_cast<char*>(&fieldName0) }, { [[$QUOTE]]some_feature[[$QUOTE]], reinterpret_cast<char*>(&fieldName1) } }">  
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

emitc.class @fooExcluded {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.field_ref = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.other_field = ["some_feature"]}
  emitc.func @"operator()"() {
    %0 = get_field @fieldName0 : !emitc.array<1xf32>
    return
  }
}

// CHECK:       emitc.class @fooExcluded {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
// CHECK-NEXT:    emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.other_field = ["some_feature"]}
// CHECK-NEXT:    emitc.field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>"> = 
// CHECK-SAME:    #emitc.opaque<"{ { [[$QUOTE]]another_feature[[$QUOTE]], reinterpret_cast<char*>(&fieldName0) } }">  
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

/// Test that the pass leaves IR unchanged if fields don't have any attributes (match failure)

emitc.class @fooNoAttrs {
  emitc.field @fieldName0 : !emitc.array<1xf32>
  emitc.func @"operator()"() {
    return
  }
}

// CHECK-NOT:     emitc.include
// CHECK-LABEL: emitc.class @fooNoAttrs {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32>
// CHECK-NEXT:    emitc.func @"operator()"() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Test that the pass leaves IR unchanged if the ClassOp doesn't have any fields (match failure)

emitc.class @fooNoFields {
  emitc.func @"operator()"() {
    return
  }
}

// CHECK-NOT:     emitc.include
// CHECK-LABEL: emitc.class @fooNoFields {
// CHECK-NEXT:    emitc.func @"operator()"() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Test that a reflection map is still created in the case that there are no
/// functions in the class

emitc.class @fooNoOperator {
  emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
}

// CHECK-LABEL: emitc.class @fooNoOperator {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
// CHECK-NEXT:    emitc.field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>"> = 
// CHECK-SAME:    #emitc.opaque<"{ { [[$QUOTE]]another_feature[[$QUOTE]], reinterpret_cast<char*>(&fieldName0) } }">
// CHECK-NEXT:    emitc.func @getBufferForName(%{{.*}}: !emitc.opaque<"std::string">) -> !emitc.ptr<!emitc.opaque<"char">> {
// CHECK-NEXT:      %[[MAP0:.*]] = get_field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>">
// CHECK-NEXT:      %[[VAL0:.*]] = member_call_opaque %[[MAP0]] "at"({{.*}}) : !emitc.opaque<"const std::map<std::string, char*>">, (!emitc.opaque<"std::string">) -> !emitc.ptr<!emitc.opaque<"char">>
// CHECK-NEXT:      return %[[VAL0]] : !emitc.ptr<!emitc.opaque<"char">>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Test that the pass returns with a match failure if a FieldOp has the specified
/// dictionary attribute with an array containing a type other than string

emitc.class @fooNonStringAttr {
  emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = [1]}
  emitc.func @"operator()"() {
    return
  }
}

// CHECK-NOT:     emitc.include
// CHECK-LABEL: emitc.class @fooNonStringAttr {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = [1]}
// CHECK-NEXT:    emitc.func @"operator()"() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

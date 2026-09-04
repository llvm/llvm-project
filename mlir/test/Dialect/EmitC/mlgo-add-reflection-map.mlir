// RUN: mlir-opt -split-input-file --mlgo-add-reflection-map="included-field-attrs=emitc.field_ref,emitc.field_ref_2" %s | FileCheck %s '-D$QUOTE=\22'

/// Tests that a reflection map is created for fields with an attribute in the
/// included-field-attrs option.

emitc.class @foo {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.field_ref = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.field_ref = ["some_feature"]}
  emitc.func @bar() {
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
// CHECK-NEXT:    emitc.func @bar() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Tests that a reflection map is created for fields in the included-field-attrs
/// option and skips fields without matching attributes.

emitc.class @foo_mixed_attrs {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.field_ref = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.other_field = ["some_feature"]}
  emitc.func @bar() {
    %0 = get_field @fieldName0 : !emitc.array<1xf32>
    return
  }
}

// CHECK:       emitc.class @foo_mixed_attrs {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
// CHECK-NEXT:    emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.other_field = ["some_feature"]}
// CHECK-NEXT:    emitc.field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>"> = 
// CHECK-SAME:    #emitc.opaque<"{ { [[$QUOTE]]another_feature[[$QUOTE]], reinterpret_cast<char*>(&fieldName0) } }">  
// CHECK-NEXT:    emitc.func @getBufferForName(%{{.*}}: !emitc.opaque<"std::string">) -> !emitc.ptr<!emitc.opaque<"char">> {
// CHECK-NEXT:      %[[MAP1:.*]] = get_field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>">
// CHECK-NEXT:      %[[VAL1:.*]] = member_call_opaque %[[MAP1]] "at"({{.*}}) : !emitc.opaque<"const std::map<std::string, char*>">, (!emitc.opaque<"std::string">) -> !emitc.ptr<!emitc.opaque<"char">>
// CHECK-NEXT:      return %[[VAL1]] : !emitc.ptr<!emitc.opaque<"char">>
// CHECK-NEXT:    }
// CHECK-NEXT:    emitc.func @bar() {
// CHECK-NEXT:      %{{.*}} = get_field @fieldName0 : !emitc.array<1xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Test that the pass bails out and leaves IR unchanged if no fields contain
/// any of the attributes specified in included-field-attrs

emitc.class @foo_unsupported_attr {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.other_field = ["another_feature"]}
  emitc.func @bar() {
    %0 = get_field @fieldName0 : !emitc.array<1xf32>
    return
  }
}

// CHECK:       emitc.class @foo_unsupported_attr {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.other_field = ["another_feature"]}
// CHECK-NEXT:    emitc.func @bar() {
// CHECK-NEXT:      %{{.*}} = get_field @fieldName0 : !emitc.array<1xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Test that the pass bails out and leaves IR unchanged if fields don't have any attributes

emitc.class @negative_foo_no_attrs {
  emitc.field @fieldName0 : !emitc.array<1xf32>
  emitc.func @bar() {
    return
  }
}

// CHECK-NOT:     emitc.include
// CHECK-LABEL: emitc.class @negative_foo_no_attrs {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32>
// CHECK-NEXT:    emitc.func @bar() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Test that the pass bails out and leaves IR unchanged if the ClassOp doesn't have any fields

emitc.class @negative_foo_no_fields {
  emitc.func @bar() {
    return
  }
}

// CHECK-NOT:     emitc.include
// CHECK-LABEL: emitc.class @negative_foo_no_fields {
// CHECK-NEXT:    emitc.func @bar() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Test that a reflection map is still created in the case that there are no
/// functions in the class

emitc.class @foo_no_operator {
  emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
}

// CHECK-LABEL: emitc.class @foo_no_operator {
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

/// Test that the pass bails out if a FieldOp has the specified dictionary attribute
/// with an array containing a type other than string

emitc.class @negative_foo_non_string_attr {
  emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = [1]}
  emitc.func @bar() {
    return
  }
}

// CHECK-NOT:     emitc.include
// CHECK-LABEL: emitc.class @negative_foo_non_string_attr {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = [1]}
// CHECK-NEXT:    emitc.func @bar() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

/// Test that the pass matches one of the multiple included attributes.

emitc.class @foo_multiple_attrs {
  emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref_2 = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.field_ref = ["some_feature"]}
  emitc.func @bar() {
    return
  }
}

// CHECK-LABEL: emitc.class @foo_multiple_attrs {
// CHECK-NEXT:    emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref_2 = ["another_feature"]}
// CHECK-NEXT:    emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.field_ref = ["some_feature"]}
// CHECK-NEXT:    emitc.field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>"> =
// CHECK-SAME:    #emitc.opaque<"{ { [[$QUOTE]]another_feature[[$QUOTE]], reinterpret_cast<char*>(&fieldName0) }, { [[$QUOTE]]some_feature[[$QUOTE]], reinterpret_cast<char*>(&fieldName1) } }">
// CHECK-NEXT:    emitc.func @getBufferForName(%{{.*}}: !emitc.opaque<"std::string">) -> !emitc.ptr<!emitc.opaque<"char">> {
// CHECK-NEXT:      %[[MAP:.*]] = get_field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>">
// CHECK-NEXT:      %[[VAL:.*]] = member_call_opaque %[[MAP]] "at"({{.*}}) : !emitc.opaque<"const std::map<std::string, char*>">, (!emitc.opaque<"std::string">) -> !emitc.ptr<!emitc.opaque<"char">>
// CHECK-NEXT:      return %[[VAL]] : !emitc.ptr<!emitc.opaque<"char">>
// CHECK-NEXT:    }
// CHECK-NEXT:    emitc.func @bar() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

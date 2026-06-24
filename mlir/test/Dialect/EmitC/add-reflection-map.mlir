// RUN: mlir-opt -split-input-file --add-reflection-map="field-attr-name=emitc.field_ref excluded-field-attrs="emitc.other_field"" %s | FileCheck %s


// Tests that a reflection map is created for fields with a certain attribute.

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
// Test that a reflection map is created for fields with a certain named attribute
// but not ones with an attribute present in the ignore-attributes option.

emitc.class @actionClass {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.field_ref = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.other_field = ["some_feature"]}
  emitc.func @"operator()"() {
    %0 = get_field @fieldName0 : !emitc.array<1xf32>
    return
  }
}

// CHECK:       emitc.class @actionClass {
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
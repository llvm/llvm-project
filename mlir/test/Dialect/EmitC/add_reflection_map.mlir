// RUN: mlir-opt --add-reflection-map="named-attribute=emitc.field_ref" %s | FileCheck %s

emitc.class @mainClass {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.field_ref = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.field_ref = ["some_feature"]}
  emitc.field @fieldName2 : !emitc.array<1xf32>  {emitc.field_ref = ["output_0"]}
  emitc.func @execute() {
  %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
  %1 = get_field @fieldName0 : !emitc.array<1xf32>
  %2 = get_field @fieldName1 : !emitc.array<1xf32>
  %3 = get_field @fieldName2 : !emitc.array<1xf32>
  %4 = subscript %2[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
  %5 = load %4 : <f32>
  %6 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
  %7 = load %6 : <f32>
  %8 = add %5, %7 : (f32, f32) -> f32
  %9 = subscript %3[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
  assign %8 : f32 to %9 : <f32>
  return
  }
}

// CHECK: module {
// CHECK-NEXT:   emitc.class @mainClass {
// CHECK-NEXT:     emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
// CHECK-NEXT:     emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.field_ref = ["some_feature"]}
// CHECK-NEXT:     emitc.field @fieldName2 : !emitc.array<1xf32> {emitc.field_ref = ["output_0"]}
// CHECK-NEXT:     emitc.func @getBufferForName(%arg0: !emitc.opaque<"std::string_view">) -> !emitc.opaque<"char"> {
// CHECK-NEXT:       %0 = "emitc.constant"() <{value = #emitc.opaque<"{  { \22another_feature\22, reinterpret_cast<char*>(&fieldName0) },  { \22some_feature\22, reinterpret_cast<char*>(&fieldName1) },  { \22output_0\22, reinterpret_cast<char*>(&fieldName2) } }">}> : () -> !emitc.opaque<"const std::map<std::string, char*>">
// CHECK-NEXT:       %1 = call_opaque "find"(%0, %arg0) : (!emitc.opaque<"const std::map<std::string, char*>">, !emitc.opaque<"std::string_view">) -> !emitc.opaque<"std::map<std::string, char*>::const_iterator">
// CHECK-NEXT:       %2 = call_opaque "end"(%0) : (!emitc.opaque<"const std::map<std::string, char*>">) -> !emitc.opaque<"std::map<std::string, char*>::const_iterator">
// CHECK-NEXT:       %3 = call_opaque "operator=="(%1, %2) : (!emitc.opaque<"std::map<std::string, char*>::const_iterator">, !emitc.opaque<"std::map<std::string, char*>::const_iterator">) -> i1
// CHECK-NEXT:       %4 = "emitc.constant"() <{value = #emitc.opaque<"nullptr">}> : () -> !emitc.opaque<"char">
// CHECK-NEXT:       %5 = call_opaque "second"(%1) : (!emitc.opaque<"std::map<std::string, char*>::const_iterator">) -> !emitc.opaque<"char">
// CHECK-NEXT:       %6 = conditional %3, %4, %5 : !emitc.opaque<"char">
// CHECK-NEXT:       return %6 : !emitc.opaque<"char">
// CHECK-NEXT:     }
// CHECK-NEXT:     emitc.func @execute() {
// CHECK-NEXT:       %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// CHECK-NEXT:       %1 = get_field @fieldName0 : !emitc.array<1xf32>
// CHECK-NEXT:       %2 = get_field @fieldName1 : !emitc.array<1xf32>
// CHECK-NEXT:       %3 = get_field @fieldName2 : !emitc.array<1xf32>
// CHECK-NEXT:       %4 = subscript %2[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-NEXT:       %5 = load %4 : <f32>
// CHECK-NEXT:       %6 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-NEXT:       %7 = load %6 : <f32>
// CHECK-NEXT:       %8 = add %5, %7 : (f32, f32) -> f32
// CHECK-NEXT:       %9 = subscript %3[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-NEXT:       assign %8 : f32 to %9 : <f32>
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
 


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
// CHECK-NEXT:   emitc.include <"map">
// CHECK-NEXT:   emitc.include <"string">
// CHECK-NEXT:   emitc.class @mainClass {
// CHECK-NEXT:     emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref = ["another_feature"]}
// CHECK-NEXT:     emitc.field @fieldName1 : !emitc.array<1xf32> {emitc.field_ref = ["some_feature"]}
// CHECK-NEXT:     emitc.field @fieldName2 : !emitc.array<1xf32> {emitc.field_ref = ["output_0"]}
// CHECK-NEXT:     emitc.field @reflectionMap : !emitc.opaque<"const std::map<std::string, char*>"> = 
// CHECK-SAME:     #emitc.opaque<"{ { \22another_feature\22, reinterpret_cast<char*>(&fieldName0) }, 
// CHECK-SAME:     { \22some_feature\22, reinterpret_cast<char*>(&fieldName1) }, 
// CHECK-SAME:     { \22output_0\22, reinterpret_cast<char*>(&fieldName2) } }">  
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
 


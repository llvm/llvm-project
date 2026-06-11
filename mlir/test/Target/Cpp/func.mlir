// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP


emitc.func @emitc_func(%arg0 : i32) {
  emitc.call_opaque "foo" (%arg0) : (i32) -> ()
  emitc.return
}
// CPP-DEFAULT: void emitc_func(int32_t [[V0:[^ ]*]]) {
// CPP-DEFAULT-NEXT: foo([[V0:[^ ]*]]);
// CPP-DEFAULT-NEXT: return;


emitc.func @return_i32() -> i32 attributes {specifiers = ["static","inline"]} {
  %0 = emitc.call_opaque "foo" (): () -> i32
  emitc.return %0 : i32
}
// CPP-DEFAULT: static inline int32_t return_i32() {
// CPP-DEFAULT-NEXT: [[V0:[^ ]*]] = foo();
// CPP-DEFAULT-NEXT: return [[V0:[^ ]*]];

// CPP-DECLTOP: static inline int32_t return_i32() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V0:]] = foo();
// CPP-DECLTOP-NEXT: return [[V0:[^ ]*]];


emitc.func @emitc_call() -> i32 {
  %0 = emitc.call @return_i32() : () -> (i32)
  emitc.return %0 : i32
}
// CPP-DEFAULT: int32_t emitc_call() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = return_i32();
// CPP-DEFAULT-NEXT: return [[V0:[^ ]*]];

// CPP-DECLTOP: int32_t emitc_call() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V0:[^ ]*]] = return_i32();
// CPP-DECLTOP-NEXT: return [[V0:[^ ]*]];

emitc.func private @extern_func(i32) attributes {specifiers = ["extern"]}
// CPP-DEFAULT: extern void extern_func(int32_t);

emitc.func private @array_arg(!emitc.array<3xi32>) attributes {specifiers = ["extern"]}
// CPP-DEFAULT: extern void array_arg(int32_t[3]);

emitc.class struct @return_i32_i32 {
  emitc.field @field0 : i32
  emitc.field @field1 : i32
}

emitc.func @return_two(%arg0: i32, %arg1: i32) -> !emitc.opaque<"struct return_i32_i32"> {
  %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>
  %1 = "emitc.member"(%0) <{member = "field0"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
  assign %arg0 : i32 to %1 : <i32>
  %2 = "emitc.member"(%0) <{member = "field1"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
  assign %arg1 : i32 to %2 : <i32>
  %3 = load %0 : <!emitc.opaque<"struct return_i32_i32">>
  return %3 : !emitc.opaque<"struct return_i32_i32">
}

emitc.func @call_two(%arg0: i32) -> i32 {
  %0 = call @return_two(%arg0, %arg0) : (i32, i32) -> !emitc.opaque<"struct return_i32_i32">
  %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>
  assign %0 : !emitc.opaque<"struct return_i32_i32"> to %1 : <!emitc.opaque<"struct return_i32_i32">>
  %2 = "emitc.member"(%1) <{member = "field1"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
  %3 = load %2 : <i32>
  return %3 : i32
}

// CPP-DEFAULT: struct return_i32_i32 {
// CPP-DEFAULT-NEXT:   int32_t field0;
// CPP-DEFAULT-NEXT:   int32_t field1;
// CPP-DEFAULT-NEXT: };
// CPP-DEFAULT-NEXT: struct return_i32_i32 return_two(int32_t [[V1:[^ ]*]], int32_t [[V2:[^ ]*]]) {
// CPP-DEFAULT-NEXT:   struct return_i32_i32 [[V3:[^ ]*]];
// CPP-DEFAULT-NEXT:   [[V3]].field0 = [[V1]];
// CPP-DEFAULT-NEXT:   [[V3]].field1 = [[V2]];
// CPP-DEFAULT-NEXT:   struct return_i32_i32 [[V4:[^ ]*]] = [[V3]];
// CPP-DEFAULT-NEXT:   return [[V4]];
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: int32_t call_two(int32_t [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT:   struct return_i32_i32 [[V2:[^ ]*]] = return_two([[V1]], [[V1]]);
// CPP-DEFAULT-NEXT:   struct return_i32_i32 [[V3:[^ ]*]];
// CPP-DEFAULT-NEXT:   [[V3]] = [[V2]];
// CPP-DEFAULT-NEXT:   int32_t [[V4:[^ ]*]] = [[V3]].field1;
// CPP-DEFAULT-NEXT:   return [[V4]];

// CPP-DECLTOP: struct return_i32_i32 {
// CPP-DECLTOP-NEXT:   int32_t field0;
// CPP-DECLTOP-NEXT:   int32_t field1;
// CPP-DECLTOP-NEXT: };
// CPP-DECLTOP-NEXT: struct return_i32_i32 return_two(int32_t [[V1:[^ ]*]], int32_t [[V2:[^ ]*]]) {
// CPP-DECLTOP-NEXT:   struct return_i32_i32 [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT:   struct return_i32_i32 [[V4:[^ ]*]];
// CPP-DECLTOP:        [[V3]].field0 = [[V1]];
// CPP-DECLTOP-NEXT:   [[V3]].field1 = [[V2]];
// CPP-DECLTOP-NEXT:   [[V4]] = [[V3]];
// CPP-DECLTOP-NEXT:   return [[V4]];
// CPP-DECLTOP-NEXT: }
// CPP-DECLTOP-NEXT: int32_t call_two(int32_t [[V1:[^ ]*]]) {
// CPP-DECLTOP-NEXT:   struct return_i32_i32 [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT:   struct return_i32_i32 [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT:   int32_t [[V4:[^ ]*]];
// CPP-DECLTOP-NEXT:   [[V2]] = return_two([[V1]], [[V1]]);
// CPP-DECLTOP:        [[V3]] = [[V2]];
// CPP-DECLTOP-NEXT:   [[V4]] = [[V3]].field1;
// CPP-DECLTOP-NEXT:   return [[V4]];

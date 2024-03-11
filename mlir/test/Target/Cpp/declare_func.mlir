// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

// CHECK: int32_t bar(int32_t [[V1:[^ ]*]]);
emitc.declare_func @bar
// CHECK: int32_t bar(int32_t [[V1:[^ ]*]]) {
emitc.func @bar(%arg0: i32) -> i32 {
    emitc.return %arg0 : i32
}


// CHECK: static inline int32_t foo(int32_t [[V1:[^ ]*]]);
emitc.declare_func @foo
// CHECK: static inline int32_t foo(int32_t [[V1:[^ ]*]]) {
emitc.func @foo(%arg0: i32) -> i32 attributes {specifiers = ["static","inline"]} {
    emitc.return %arg0 : i32
}


// CHECK: void array_arg(int32_t [[V2:[^ ]*]][3]);
emitc.declare_func @array_arg
// CHECK: void array_arg(int32_t  [[V2:[^ ]*]][3]) {
emitc.func @array_arg(%arg0: !emitc.array<3xi32>) {
    emitc.return
}

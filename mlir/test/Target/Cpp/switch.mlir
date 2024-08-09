// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s

func.func @switch_func(%a: i32, %b: i32, %c: i32) -> () {
    cf.switch %b : i32, [
    default: ^bb1(%a : i32),
    42: ^bb1(%b : i32),
    43: ^bb2(%c : i32),
    44: ^bb3(%c : i32)
    ]

    ^bb1(%x1 : i32) :
        %y1 = "emitc.add" (%x1, %x1) : (i32, i32) -> i32
        return

    ^bb2(%x2 : i32) :
        %y2 = "emitc.sub" (%x2, %x2) : (i32, i32) -> i32
        return

    ^bb3(%x3 : i32) :
        %y3 = "emitc.mul" (%x3, %x3) : (i32, i32) -> i32
        return
}
// CHECK: void switch_func(int32_t [[V0:[^ ]*]], int32_t [[V1:[^ ]*]], int32_t [[V2:[^ ]*]]) {
// CHECK: switch([[V1:[^ ]*]]) {
// CHECK-NEXT: case (42): {
// CHECK-NEXT: v7 = v2;
// CHECK-NEXT: goto label2;
// CHECK-NEXT: }
// CHECK-NEXT: case (43): {
// CHECK-NEXT: v8 = v3;
// CHECK-NEXT: goto label3;
// CHECK-NEXT: }
// CHECK-NEXT: case (44): {
// CHECK-NEXT: v9 = v3;
// CHECK-NEXT: goto label4;
// CHECK-NEXT: }
// CHECK-NEXT: default: {
// CHECK-NEXT: v7 = v1;
// CHECK-NEXT: goto label2;
// CHECK-NEXT: }
// CHECK-NEXT: }

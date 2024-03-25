// RUN: mlir-opt %s -test-distinct-attrs | FileCheck %s

// CHECK: #[[DIST0:.*]] = distinct[0]<42 : i32>
// CHECK: #[[DIST1:.*]] = distinct[1]<42 : i32>
#distinct = distinct[0]<42 : i32>
// CHECK: #[[DIST2:.*]] = distinct[2]<42 : i32>
// CHECK: #[[DIST3:.*]] = distinct[3]<42 : i32>
#distinct1 = distinct[1]<42 : i32>
// CHECK: #[[DIST4:.*]] = distinct[4]<43 : i32>
// CHECK: #[[DIST5:.*]] = distinct[5]<43 : i32>
#distinct2 = distinct[2]<43 : i32>
// CHECK: #[[DIST6:.*]] = distinct[6]<@foo_1>
// CHECK: #[[DIST7:.*]] = distinct[7]<@foo_1>
#distinct3 = distinct[3]<@foo_1>

// Copies made for foo_2
// CHECK: #[[DIST8:.*]] = distinct[8]<42 : i32>
// CHECK: #[[DIST9:.*]] = distinct[9]<42 : i32>
// CHECK: #[[DIST10:.*]] = distinct[10]<43 : i32>
// CHECK: #[[DIST11:.*]] = distinct[11]<@foo_1>

// Copies made for foo_3
// CHECK: #[[DIST12:.*]] = distinct[12]<42 : i32>
// CHECK: #[[DIST13:.*]] = distinct[13]<42 : i32>
// CHECK: #[[DIST14:.*]] = distinct[14]<43 : i32>
// CHECK: #[[DIST15:.*]] = distinct[15]<@foo_1>

// Copies made for foo_4
// CHECK: #[[DIST16:.*]] = distinct[16]<42 : i32>
// CHECK: #[[DIST17:.*]] = distinct[17]<42 : i32>
// CHECK: #[[DIST18:.*]] = distinct[18]<43 : i32>
// CHECK: #[[DIST19:.*]] = distinct[19]<@foo_1>

// CHECK: @foo_1
func.func @foo_1() {
  // CHECK: "test.op"() {distinct.input = #[[DIST0]], distinct.output = #[[DIST1]]}
  "test.op"() {distinct.input = #distinct} : () -> ()
  // CHECK: "test.op"() {distinct.input = #[[DIST2]], distinct.output = #[[DIST3]]}
  "test.op"() {distinct.input = #distinct1} : () -> ()
  // CHECK: "test.op"() {distinct.input = #[[DIST4]], distinct.output = #[[DIST5]]}
  "test.op"() {distinct.input = #distinct2} : () -> ()
  // CHECK: "test.op"() {distinct.input = #[[DIST6]], distinct.output = #[[DIST7]]}
  "test.op"() {distinct.input = #distinct3} : () -> ()
}

func.func @foo_2() {
  // CHECK: "test.op"() {distinct.input = #[[DIST0]], distinct.output = #[[DIST8]]}
  "test.op"() {distinct.input = #distinct} : () -> ()
  // CHECK: "test.op"() {distinct.input = #[[DIST2]], distinct.output = #[[DIST9]]}
  "test.op"() {distinct.input = #distinct1} : () -> ()
  // CHECK: "test.op"() {distinct.input = #[[DIST4]], distinct.output = #[[DIST10]]}
  "test.op"() {distinct.input = #distinct2} : () -> ()
  // CHECK: "test.op"() {distinct.input = #[[DIST6]], distinct.output = #[[DIST11]]}
  "test.op"() {distinct.input = #distinct3} : () -> ()
}

func.func @foo_3() {
  // CHECK: "test.op"() {distinct.input = #[[DIST0]], distinct.output = #[[DIST12]]}
  "test.op"() {distinct.input = #distinct} : () -> ()
  // CHECK: "test.op"() {distinct.input = #[[DIST2]], distinct.output = #[[DIST13]]}
  "test.op"() {distinct.input = #distinct1} : () -> ()
  // CHECK: "test.op"() {distinct.input = #[[DIST4]], distinct.output = #[[DIST14]]}
  "test.op"() {distinct.input = #distinct2} : () -> ()
  // CHECK: "test.op"() {distinct.input = #[[DIST6]], distinct.output = #[[DIST15]]}
  "test.op"() {distinct.input = #distinct3} : () -> ()
}

func.func @foo_4() {
  // CHECK: "test.op"() {distinct.input = #[[DIST0]], distinct.output = #[[DIST16]]}
  "test.op"() {distinct.input = #distinct} : () -> ()
  // CHECK: "test.op"() {distinct.input = #[[DIST2]], distinct.output = #[[DIST17]]}
  "test.op"() {distinct.input = #distinct1} : () -> ()
  // CHECK: "test.op"() {distinct.input = #[[DIST4]], distinct.output = #[[DIST18]]}
  "test.op"() {distinct.input = #distinct2} : () -> ()
  // CHECK: "test.op"() {distinct.input = #[[DIST6]], distinct.output = #[[DIST19]]}
  "test.op"() {distinct.input = #distinct3} : () -> ()
}

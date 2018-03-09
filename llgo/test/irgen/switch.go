// RUN: llgo -S -emit-llvm -o - %s | FileCheck %s

package foo

// CHECK: switch i32
// CHECK-NEXT: i32 0, label %[[L0:.*]]
// CHECK-NEXT: i32 1, label %[[L1:.*]]
// CHECK-NEXT: i32 2, label %[[L2:.*]]
// CHECK-NEXT: ]
// CHECK: [[L0]]:
// CHECK-NEXT: ret i32 1
// CHECK: [[L1]]:
// CHECK-NEXT: ret i32 2
// CHECK: [[L2]]:
// CHECK-NEXT: ret i32 0
func F1(x int32) int32 {
	switch x {
	case 0:
		return 1
	case 1:
		return 2
	case 2:
		return 0
	}
	panic("unreachable")
}

// CHECK: switch i64
// CHECK-NEXT: i64 0
// CHECK-NEXT: ]
// CHECK: icmp eq i64 {{.*}}, 1
func F2(x int64) bool {
	return x == 0 || x == 0 || x == 1
}

// CHECK: switch i64
// CHECK-NEXT: i64 0
// CHECK-NEXT: ]
func F3(x int64) bool {
	return x == 0 || x == 0 || x == 0
}

// CHECK: switch i64
// CHECK-NEXT: i64 0
// CHECK-NEXT: i64 1
// CHECK-NEXT: i64 2
// CHECK-NEXT: ]
// CHECK: icmp eq i64 {{.*}}, 3
func F4(x int64) bool {
	return x == 0 || x == 1 || x == 2 || x == 3
}

// CHECK-NOT: switch double
func F5(x float64) float64 {
	switch x {
	case 0:
		return 1.0
	case 1.0:
		return 0
	}
	panic("unreachable")
}

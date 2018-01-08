// RUN: llgo -S -emit-llvm -o - %s | FileCheck %s

package foo

func F() {
	// CHECK: call void @__go_panic
	// CHECK-NEXT: ret void
	// CHECK-NEXT: }
	go panic("")
}

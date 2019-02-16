// RUN: llgo -S -emit-llvm -o - %s | FileCheck %s

package foo

// CHECK-NOT: alloca [1024 x i8]
// CHECK-NOT: alloca [2048 x i8]
// CHECK: alloca [4096 x i8]
func F() {
	ch1 := make(chan [1024]byte)
	ch2 := make(chan [2048]byte)
	ch3 := make(chan [4096]byte)
	select {
	case <-ch1:
	case _ = <-ch2:
	case x := <-ch3:
		_ = x[0]
	}
}

// RUN: llgo -S -emit-llvm -o - %s | FileCheck %s

package foo

type X struct {
	indices [1]int
}

// CHECK-NOT: load [200 x i64]
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}}, i64 160000, i32 1, i1 false)
var _ = [100][200]int{}[0][0]

// CHECK-NOT: load [1024 x i64]
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}}, i64 8192, i32 1, i1 false)
var _ = [1024]int{}[X{}.indices[0]]

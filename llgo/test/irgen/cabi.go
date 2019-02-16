// RUN: llgo -S -emit-llvm -o - %s | FileCheck %s

package foo

// CHECK: define void @foo.Test01_SI8(i8* nest, i8 signext)
func Test01_SI8(x int8) {}
// CHECK: define void @foo.Test02_UI8(i8* nest, i8 zeroext)
func Test02_UI8(x uint8) {}

// CHECK: define void @foo.Test03_SI16(i8* nest, i16 signext)
func Test03_SI16(x int16) {}
// CHECK: define void @foo.Test04_UI16(i8* nest, i16 zeroext)
func Test04_UI16(x uint16) {}

// CHECK: define void @foo.Test05_SI32(i8* nest, i32)
func Test05_SI32(x int32) {}
// CHECK: define void @foo.Test06_UI32(i8* nest, i32)
func Test06_UI32(x uint32) {}

// CHECK: define void @foo.Test07_SI64(i8* nest, i64)
func Test07_SI64(x int64) {}
// CHECK: define void @foo.Test08_UI64(i8* nest, i64)
func Test08_UI64(x uint64) {}

// RUN: llgo -fgo-pkgpath=p -c -o %T/p.o %S/Inputs/mangling-synthetic-p.go
// RUN: llgo -fgo-pkgpath=q -I %T -S -emit-llvm -o - %s | FileCheck %s

package q

import "p"

// CHECK-DAG: define linkonce_odr void @p.f.N3_q.T(i8* nest, i8*)
// CHECK-DAG: define linkonce_odr void @p.f.pN3_q.T(i8* nest, i8*)
type T struct { p.U }

// CHECK-DAG: declare void @q.f.N3_q.T(i8* nest, i8*)
// CHECK-DAG: define linkonce_odr void @q.f.pN3_q.T(i8* nest, i8*)
func (T) f()

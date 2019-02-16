// RUN: llgo -O0 -S -o - %s | FileCheck %s

package gotest

// CHECK-NOT: deadfunc
func deadfunc()

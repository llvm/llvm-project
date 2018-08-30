// RUN: llgo -fgo-pkgpath=llvm.org/llvm -S -emit-llvm -o - %s | FileCheck %s

package llvm

// CHECK: @llvm_org_llvm.F
func F() {
}

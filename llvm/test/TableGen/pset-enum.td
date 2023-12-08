// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s | FileCheck %s
// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common -DUSE_NAMESPACE %s | FileCheck --check-prefixes=CHECK,NAMESPACE %s

include "reg-with-subregs-common.td"

// CHECK-LABEL:    // Register pressure sets enum.
// NAMESPACE-NEXT: namespace TestNamespace {
// CHECK-NEXT:     enum RegisterPressureSets {
// CHECK-NEXT:       GPR32 = 0,
// CHECK-NEXT:     };
// NAMESPACE-NEXT: } // end namespace TestNamespace

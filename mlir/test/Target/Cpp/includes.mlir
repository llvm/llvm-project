// RUN: mlir-translate --mlir-to-cpp %s | FileCheck %s --check-prefix=DEFAULT
// RUN: mlir-translate --mlir-to-cpp --emit-includes=true %s | FileCheck %s --check-prefix=DEFAULT
// RUN: mlir-translate --mlir-to-cpp --emit-includes=false %s | FileCheck %s --check-prefix=NO-INCLUDES

// DEFAULT:      #include "local.h"
// DEFAULT-NEXT: #include <vector>
// DEFAULT:      void test_include() {
// DEFAULT-NEXT:   #include <cstdint>
// DEFAULT-NEXT:   return;

// NO-INCLUDES-NOT:  #include
// NO-INCLUDES:      void test_include() {
// NO-INCLUDES-NEXT:   return;
// NO-INCLUDES-NOT:    #include

emitc.include "local.h"
emitc.include <"vector">

func.func @test_include() {
  emitc.include <"cstdint">
  return
}

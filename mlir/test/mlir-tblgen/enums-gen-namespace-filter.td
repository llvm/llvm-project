// RUN: mlir-tblgen -gen-enum-decls -enum-namespace=test::ns1 -I %S/../../include %s | FileCheck %s --check-prefix=NS1
// RUN: mlir-tblgen -gen-enum-decls -enum-namespace=test::ns2 -I %S/../../include %s | FileCheck %s --check-prefix=NS2
// RUN: mlir-tblgen -gen-enum-decls -enum-namespace=test::ns -I %S/../../include %s | FileCheck %s --check-prefix=NS
// RUN: mlir-tblgen -gen-enum-decls -enum-namespace=test -I %S/../../include %s | FileCheck %s --check-prefix=TEST-NS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def EnumNS1 : I32BitEnumAttr<"EnumNS1", "", []> {
  let genSpecializedAttr = 0;
  let cppNamespace = "test::ns1";
}

def EnumNS2 : I32BitEnumAttr<"EnumNS2", "", []> {
  let genSpecializedAttr = 0;
  let cppNamespace = "test::ns2";
}

// NS1-NOT: enum class EnumNS2
// NS1: enum class EnumNS1
// NS1-NOT: enum class EnumNS2

// NS2-NOT: enum class EnumNS1
// NS2: enum class EnumNS2

// NS-NOT: enum class EnumNS1
// NS-NOT: enum class EnumNS2

// TEST-NS: enum class EnumNS1
// TEST-NS: enum class EnumNS2

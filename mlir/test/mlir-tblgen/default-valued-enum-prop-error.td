// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include %s 2>&1 | FileCheck %s

include "mlir/IR/EnumAttr.td"

def OtherCase : I32EnumCase<"Other", 0>;
def TestEnum : I32Enum<"TestEnum", "", [I32EnumCase<"First", 0>]>;

// CHECK: error: assertion failed: default `Other` is not a case of {{.*}}TestEnum
def BadDefault : DefaultValuedEnumProp<TestEnum, OtherCase>;

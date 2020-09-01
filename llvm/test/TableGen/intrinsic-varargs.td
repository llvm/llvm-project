// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s | FileCheck %s
// XFAIL: vg_leak

include "llvm/CodeGen/ValueTypes.td"

class IntrinsicProperty<bit is_default = 0> {
  bit IsDefault = is_default;
}
class SDNodeProperty;

class LLVMType<ValueType vt> {
  ValueType VT = vt;
}

class Intrinsic<string name, list<LLVMType> param_types = []> {
  string LLVMName = name;
  bit isTarget = 0;
  string TargetPrefix = "";
  list<LLVMType> RetTypes = [];
  list<LLVMType> ParamTypes = param_types;
  list<IntrinsicProperty> IntrProperties = [];
  list<SDNodeProperty> Properties = [];
  bit DisableDefaultAttributes = 1;
}

def llvm_vararg_ty : LLVMType<isVoid>;   // this means vararg here

// CHECK: /* 0 */ 0, 29, 0,
def int_foo : Intrinsic<"llvm.foo", [llvm_vararg_ty]>;

// RUN: llvm-tblgen -gen-intrinsic %s | FileCheck %s
// XFAIL: vg_leak

class IntrinsicProperty;

class ValueType<int size, int value> {
  string Namespace = "MVT";
  int Size = size;
  int Value = value;
}

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
}

// isVoid needs to match the definition in ValueTypes.td
def isVoid : ValueType<0, 108>;   // Produces no value
def llvm_vararg_ty : LLVMType<isVoid>;   // this means vararg here

// CHECK: /* 0 */ 0, 29, 0,
def int_foo : Intrinsic<"llvm.foo", [llvm_vararg_ty]>;

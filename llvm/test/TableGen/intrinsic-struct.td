// RUN: llvm-tblgen -gen-intrinsic-enums %s | FileCheck %s
// XFAIL: vg_leak

class IntrinsicProperty;
class SDNodeProperty;

class ValueType<int size, int value> {
  string Namespace = "MVT";
  int Size = size;
  int Value = value;
}

class LLVMType<ValueType vt> {
  ValueType VT = vt;
}

class Intrinsic<string name, list<LLVMType> ret_types = []> {
  string LLVMName = name;
  bit isTarget = 0;
  string TargetPrefix = "";
  list<LLVMType> RetTypes = ret_types;
  list<LLVMType> ParamTypes = [];
  list<IntrinsicProperty> IntrProperties = [];
  list<SDNodeProperty> Properties = [];
}

def iAny : ValueType<0, 253>;
def llvm_anyint_ty : LLVMType<iAny>;

// Make sure we can return up to 8 values
// CHECK: returns_8_results = {{[0-9]+}}, // llvm.returns.8.results
def int_returns_8_results : Intrinsic<"llvm.returns.8.results",
    [llvm_anyint_ty, llvm_anyint_ty, llvm_anyint_ty, llvm_anyint_ty,
     llvm_anyint_ty, llvm_anyint_ty, llvm_anyint_ty, llvm_anyint_ty]>;

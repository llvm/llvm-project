// RUN: llvm-tblgen -gen-intrinsic-enums %s | FileCheck %s
// XFAIL: vg_leak

class IntrinsicProperty<bit is_default = 0> {
  bit IsDefault = is_default;
}

class SDNodeProperty;

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
  list<SDNodeProperty> Properties = [];
  bit DisableDefaultAttributes = 1;
}

def iAny : ValueType<0, 253>;
def llvm_anyint_ty : LLVMType<iAny>;

// Make sure we generate the long name without crashing
// CHECK: this_is_a_really_long_intrinsic_name_but_we_should_still_not_crash,  // llvm.this.is.a.really.long.intrinsic.name.but.we.should.still.not.crash
def int_foo : Intrinsic<"llvm.foo", [llvm_anyint_ty]>;
def int_this_is_a_really_long_intrinsic_name_but_we_should_still_not_crash : Intrinsic<"llvm.this.is.a.really.long.intrinsic.name.but.we.should.still.not.crash", [llvm_anyint_ty]>;


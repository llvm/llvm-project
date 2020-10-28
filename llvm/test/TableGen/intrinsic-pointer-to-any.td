// RUN: llvm-tblgen -gen-intrinsic-impl %s | FileCheck %s

// This test is validating that it an Intrinsic with an LLVMPointerType to
// llvm_any_ty still properly work after r363233. That patch rewrote the
// substitution handling code in the Intrinsic Emitter, and didn't consider this
// case, so TableGen would hit an assertion in EncodeFixedType that was checking
// to ensure that the substitution being processed was correctly replaced.

class IntrinsicProperty<bit is_default = 0> {
  bit IsDefault = is_default;
}

class SDNodeProperty;

class ValueType<int size, int value> {
  string Namespace = "MVT";
  int Size = size;
  int Value = value;
}

def iPTR   : ValueType<0  , 254>;
def Any    : ValueType<0  , 255>;

class LLVMType<ValueType vt> {
  ValueType VT = vt;
  int isAny = 0;
}


class Intrinsic<list<LLVMType> ret_types> {
  string LLVMName = "";
  string TargetPrefix = "";   // Set to a prefix for target-specific intrinsics.
  list<LLVMType> RetTypes = ret_types;
  list<LLVMType> ParamTypes = [];
  list<IntrinsicProperty> IntrProperties = [];
  list<SDNodeProperty> Properties = [];
  bit isTarget = 0;
  bit DisableDefaultAttributes = 1;
}

class LLVMQualPointerType<LLVMType elty>
  : LLVMType<iPTR>{
  LLVMType ElTy = elty;
  int AddrSpace = 0;
}

class LLVMPointerType<LLVMType elty>
  : LLVMQualPointerType<elty>;

let isAny = 1 in {
  def llvm_any_ty        : LLVMType<Any>;
}
def i8 : ValueType<8, 3>;
def llvm_i8_ty : LLVMType<i8>;

def int_has_ptr_to_any : Intrinsic<[LLVMPointerType<llvm_any_ty>, llvm_i8_ty]>;
// CHECK: /* 0 */ 21, 14, 15, 0, 2, 0

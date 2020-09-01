// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s | FileCheck %s

// Get the minimum blurb necessary to process ...
include "llvm/CodeGen/ValueTypes.td"
include "llvm/CodeGen/SDNodeProperties.td"

class LLVMType<ValueType vt> {
  ValueType VT = vt;
  int isAny = 0;
}

def llvm_i32_ty        : LLVMType<i32>;

class IntrinsicProperty<bit is_default = 0> {
  bit IsDefault = is_default;
}

def IntrNoMem : IntrinsicProperty;
def IntrHasSideEffects : IntrinsicProperty;


class Intrinsic<list<LLVMType> ret_types,
                list<LLVMType> param_types = [],
                list<IntrinsicProperty> intr_properties = [],
                string name = "",
                list<SDNodeProperty> sd_properties = []> : SDPatternOperator {
  string LLVMName = name;
  string TargetPrefix = "";
  list<LLVMType> RetTypes = ret_types;
  list<LLVMType> ParamTypes = param_types;
  list<IntrinsicProperty> IntrProperties = intr_properties;
  let Properties = sd_properties;
  bit DisableDefaultAttributes = 1;


  bit isTarget = 0;
}

// ... this intrinsic.
def int_random_gen   : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrHasSideEffects]>;

// CHECK: 1, // llvm.random.gen
// CHECK: case 1:
// CHECK-NEXT: Atts[] = {Attribute::NoUnwind}

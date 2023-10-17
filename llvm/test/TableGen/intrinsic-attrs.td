// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s | FileCheck %s

// Get the minimum blurb necessary to process ...
include "llvm/CodeGen/ValueTypes.td"
include "llvm/CodeGen/SDNodeProperties.td"

class LLVMType<ValueType vt> {
  ValueType VT = vt;
  int isAny = 0;
}

def llvm_i32_ty     : LLVMType<i32>;
def llvm_ptr_ty     : LLVMType<iPTR>;

class AttrIndex<int idx> {
  int Value = idx;
}

def FuncIndex : AttrIndex<-1>;
def RetIndex : AttrIndex<0>;
class ArgIndex<int argNo> : AttrIndex<!add(argNo, 1)>;

class IntrinsicProperty<bit is_default = 0> {
  bit IsDefault = is_default;
}

def IntrNoMem : IntrinsicProperty;
def IntrHasSideEffects : IntrinsicProperty;
class Dereferenceable<AttrIndex idx, int bytes> : IntrinsicProperty {
  int ArgNo = idx.Value;
  int Bytes = bytes;
}

class Intrinsic<list<LLVMType> ret_types,
                list<LLVMType> param_types = [],
                list<IntrinsicProperty> intr_properties = [],
                string name = "",
                list<SDNodeProperty> sd_properties = [],
                bit disable_default_attributes = 0> : SDPatternOperator {
  string LLVMName = name;
  string TargetPrefix = "";
  list<LLVMType> RetTypes = ret_types;
  list<LLVMType> ParamTypes = param_types;
  list<IntrinsicProperty> IntrProperties = intr_properties;
  let Properties = sd_properties;
  bit DisableDefaultAttributes = 1;


  bit isTarget = 0;
  bit DisableDefaultAttributes = disable_default_attributes;
}

// ... this intrinsic.
def int_random_gen   : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrHasSideEffects]>;

def int_deref_ptr_ret : Intrinsic<[llvm_ptr_ty], [], [Dereferenceable<RetIndex, 16>]>;

// CHECK: static AttributeSet getIntrinsicArgAttributeSet(LLVMContext &C, unsigned ID) {
// CHECK-NEXT:   switch (ID) {
// CHECK-NEXT: default: llvm_unreachable("Invalid attribute set number");
// CHECK-NEXT: case 0:
// CHECK-NEXT:     return AttributeSet::get(C, {
// CHECK-NEXT: Attribute::get(C, Attribute::Dereferenceable, 16),
// CHECK-NEXT: });
// CHECK-NEXT: }
// CHECK-NEXT: }

// CHECK: static AttributeSet getIntrinsicFnAttributeSet(
// CHECK: case 0:
// CHECK-NEXT: return AttributeSet::get(C, {
// CHECK-NEXT: Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT: });


// CHECK: 1, // llvm.deref.ptr.ret
// CHECK: 2, // llvm.random.gen

// CHECK: case 1:
// CHECK-NEXT: AS[0] = {0, getIntrinsicArgAttributeSet(C, 0)};
// CHECK-NEXT: AS[1] = {AttributeList::FunctionIndex, getIntrinsicFnAttributeSet(C, 0)};
// CHECK-NEXT: NumAttrs = 2;

// CHECK: case 2:
// CHECK-NEXT: AS[0] = {AttributeList::FunctionIndex, getIntrinsicFnAttributeSet(C, 1)};
// CHECK-NEXT: NumAttrs = 1;

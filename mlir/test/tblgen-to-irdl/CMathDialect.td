// RUN: tblgen-to-irdl %s -I=%S/../../include --gen-dialect-irdl-defs --dialect=cmath | FileCheck %s

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

// CHECK-LABEL: irdl.dialect @cmath {
def CMath_Dialect : Dialect {
  let name = "cmath";
}

class CMath_Type<string name, string typeMnemonic, list<Trait> traits = []>
: TypeDef<CMath_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class CMath_Op<string mnemonic, list<Trait> traits = []>
    : Op<CMath_Dialect, mnemonic, traits>;

def f32Orf64Type : Or<[CPred<"::llvm::isa<::mlir::F32>">,
                       CPred<"::llvm::isa<::mlir::F64>">]>;

def CMath_ComplexType : CMath_Type<"ComplexType", "complex"> {
  let parameters = (ins f32Orf64Type:$elementType);
}

// CHECK:      irdl.operation @identity {
// CHECK-NEXT:   %0 = irdl.base "!cmath.complex"
// CHECK-NEXT:   irdl.operands()
// CHECK-NEXT:   irdl.results(%0)
// CHECK-NEXT: }
def CMath_IdentityOp : CMath_Op<"identity"> {
  let results = (outs CMath_ComplexType:$out);
}

// CHECK:      irdl.operation @mul {
// CHECK-NEXT:   %0 = irdl.base "!cmath.complex"
// CHECK-NEXT:   %1 = irdl.base "!cmath.complex"
// CHECK-NEXT:   %2 = irdl.base "!cmath.complex"
// CHECK-NEXT:   irdl.operands(%0, %1)
// CHECK-NEXT:   irdl.results(%2)
// CHECK-NEXT: }
def CMath_MulOp : CMath_Op<"mul"> {
  let arguments = (ins CMath_ComplexType:$in1, CMath_ComplexType:$in2);
  let results = (outs CMath_ComplexType:$out);
}

// CHECK:      irdl.operation @norm {
// CHECK-NEXT:   %0 = irdl.any
// CHECK-NEXT:   %1 = irdl.base "!cmath.complex"
// CHECK-NEXT:   irdl.operands(%0)
// CHECK-NEXT:   irdl.results(%1)
// CHECK-NEXT: }
def CMath_NormOp : CMath_Op<"norm"> {
  let arguments = (ins AnyType:$in);
  let results = (outs CMath_ComplexType:$out);
}

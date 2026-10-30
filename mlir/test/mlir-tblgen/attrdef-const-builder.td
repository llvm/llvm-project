// RUN: mlir-tblgen -print-records -I %S/../../include %s | FileCheck %s --check-prefix=RECORD
// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s --check-prefix=OP

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

class TestAttr<string name> : AttrDef<Test_Dialect, name> {
  let hasCustomAssemblyFormat = 1;
}

// An attribute is its own constant value, so it is const-buildable.
// RECORD-LABEL: def AttrNoParams
// RECORD: string constBuilderCall = "$0";
def AttrNoParams : TestAttr<"NoParams"> {
  let mnemonic = "no_params";
}

// Unless it returns a different C++ type than the attribute class.
// RECORD-LABEL: def AttrOtherReturnType
// RECORD: string constBuilderCall = ?;
def AttrOtherReturnType : TestAttr<"OtherReturnType"> {
  let mnemonic = "other_return_type";
  let returnType = "bool";
  let convertFromStorage = "$_self != nullptr";
}

def OpWithDefaultValuedAttrDef : Op<Test_Dialect, "op", []> {
  let arguments = (ins
    DefaultValuedAttr<AttrNoParams, "$_builder.getAttr<::test::NoParamsAttr>()">:$attr
  );
}

// OP-LABEL: void OpWithDefaultValuedAttrDef::populateDefaultProperties
// OP: if (!properties.attr)
// OP-NEXT: properties.attr = odsBuilder.getAttr<::test::NoParamsAttr>();

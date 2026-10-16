//===- WasmSSATypes.td - WasmSSA types def ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WasmSSA_TYPES
#define WasmSSA_TYPES

include "mlir/Dialect/WasmSSA/IR/WasmSSABase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"

def WasmSSA_IntegerType : AnyTypeOf<[I32, I64]>;
def WasmSSA_FPType: AnyTypeOf<[F32, F64]>;
def WasmSSA_NumericType : AnyTypeOf<[WasmSSA_IntegerType, WasmSSA_FPType]>{
  let cppFunctionName = "isWasmNumericType";
}
def WasmSSA_VecType : AnyTypeOf<[I128]>;

class WasmSSA_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<WasmSSA_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def WasmSSA_FuncRefType : WasmSSA_Type<"FuncRef", "funcref"> {
  let summary = "Opaque type for function reference";
  let assemblyFormat = "";
}

def WasmSSA_ExternRefType : WasmSSA_Type<"ExternRef", "externref"> {
  let summary = "Opaque type for external reference";
}

def WasmSSA_RefType : AnyTypeOf<[WasmSSA_FuncRefType, WasmSSA_ExternRefType]> {
  let cppFunctionName = "isWasmRefType";
}

def WasmSSA_ValType : AnyTypeOf<[WasmSSA_NumericType, WasmSSA_VecType, WasmSSA_RefType]> {
  let cppFunctionName = "isWasmValueType";
}

def WasmSSA_ResultType : TupleOf<[WasmSSA_ValType]>;

def WasmSSA_FuncType : TypeAlias<FunctionType>;

def WasmSSA_LimitType : WasmSSA_Type<"Limit", "limit"> {
  let summary = "Wasm limit type";
  let parameters = (ins "uint32_t":$min,
                        OptionalParameter<"std::optional<uint32_t>">:$max);
  let assemblyFormat = "`[` $min `` `:` ($max^)? `]`";
}

def WasmSSA_LocalRef : WasmSSA_Type<"LocalRef", "local"> {
  let summary = "Type of a local variable";
  let parameters = (ins WasmSSA_ValType: $elementType);
  let assemblyFormat = "`ref` `to` $elementType";
  let builders = [TypeBuilderWithInferredContext<(ins "Type":$typeParam), [{
      return get(typeParam.getContext(), typeParam);
    }]>,];

}

def WasmSSA_TableType : WasmSSA_Type<"Table", "tabletype"> {
  let summary = "Wasm table type";
  let parameters = (ins WasmSSA_RefType:$reference,
                       WasmSSA_LimitType:$limit);
  let assemblyFormat = "$reference $limit";
}

def WasmSSA_FuncTypeAttr : TypeAttrOf<WasmSSA_FuncType>;
def WasmSSA_LimitTypeAttr : TypeAttrOf<WasmSSA_LimitType>;
def WasmSSA_TableTypeAttr : TypeAttrOf<WasmSSA_TableType>;
def WasmSSA_ValTypeAttr : TypeAttrOf<WasmSSA_ValType>;

def WasmSSA_IntegerAttr : AnyAttrOf<[I32Attr, I64Attr]>;
def WasmSSA_FPAttr : AnyAttrOf<[F32Attr, F64Attr]>;
def WasmSSA_NumericAttr : AnyAttrOf<[WasmSSA_IntegerAttr, WasmSSA_FPAttr]>;

#endif // WasmSSA_TYPES

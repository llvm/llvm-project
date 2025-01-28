//===-- QuantBytecode.td - Quant bytecode defs -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the Quant bytecode reader/writer definition file.
//
//===----------------------------------------------------------------------===//

#ifndef QUANT_BYTECODE
#define QUANT_BYTECODE

include "mlir/IR/BytecodeBase.td"

def DoubleAPFloat:
  WithParser <"succeeded(readDoubleAPFloat($_reader, $_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeAPFloatWithKnownSemantics(APFloat($_getter))",
  WithType   <"double">>>>;
def DoubleAPFloatList : List<DoubleAPFloat>;

let cType = "AnyQuantizedType" in {

def AnyQuantizedType: DialectType<(type
  VarInt:$flags,
  Type:$storageType,
  SignedVarInt:$storageTypeMin,
  SignedVarInt:$storageTypeMax
)> {
  let printerPredicate = "!$_val.getExpressedType()";
  let cBuilder = [{
    get<$_resultType>(context, flags, storageType, nullptr,
      storageTypeMin, storageTypeMax)
  }];
}

def AnyQuantizedTypeWithExpressedType: DialectType<(type
  VarInt:$flags,
  Type:$storageType,
  Type:$expressedType,
  SignedVarInt:$storageTypeMin,
  SignedVarInt:$storageTypeMax
)> {
  let printerPredicate = "!!$_val.getExpressedType()";
}
}

def CalibratedQuantizedType: DialectType<(type
  Type:$expressedType,
  DoubleAPFloat:$min,
  DoubleAPFloat:$max
)>;

def UniformQuantizedType: DialectType<(type
  VarInt:$flags,
  Type:$storageType,
  Type:$expressedType,
  DoubleAPFloat:$scale,
  SignedVarInt:$zeroPoint,
  SignedVarInt:$storageTypeMin,
  SignedVarInt:$storageTypeMax
)>;

def UniformQuantizedPerAxisType: DialectType<(type
  VarInt:$flags,
  Type:$storageType,
  Type:$expressedType,
  VarInt:$quantizedDimension,
  SignedVarInt:$storageTypeMin,
  SignedVarInt:$storageTypeMax,
  Array<DoubleAPFloatList>:$scales,
  Array<SignedVarIntList>:$zeroPoints
)> {
  // Note: builder order differs from bytecode.
  let cBuilder = [{
      get<$_resultType>(context, flags, storageType, expressedType, scales,
        zeroPoints, quantizedDimension, storageTypeMin, storageTypeMax)
  }];
}

/// This enum contains marker codes used to indicate which attribute is
/// currently being decoded, and how it should be decoded. The order of these
/// codes should generally be unchanged, as any changes will inevitably break
/// compatibility with older bytecode.

def QuantDialectTypes : DialectTypes<"Quant"> {
  let elems = [
    ReservedOrDead,
    AnyQuantizedType,
    AnyQuantizedTypeWithExpressedType,
    CalibratedQuantizedType,
    UniformQuantizedType,
    UniformQuantizedPerAxisType
  ];
}

#endif // QUANT_BYTECODE
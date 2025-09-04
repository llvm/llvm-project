//===- IRTypes.h - Type Interfaces ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_IRTYPES_H
#define MLIR_BINDINGS_PYTHON_IRTYPES_H

#include "mlir-c/BuiltinTypes.h"
#include "mlir/Bindings/Python/IRModule.h"

namespace mlir::python {

/// Shaped Type Interface - ShapedType
class PyShapedType : public PyConcreteType<PyShapedType> {
public:
  static const IsAFunctionTy isaFunction;
  static constexpr const char *pyClassName = "ShapedType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);

private:
  void requireHasRank();
};

class PyIntegerType : public PyConcreteType<PyIntegerType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAInteger;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirIntegerTypeGetTypeID;
  static constexpr const char *pyClassName = "IntegerType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Index Type subclass - IndexType.
class PyIndexType : public PyConcreteType<PyIndexType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAIndex;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirIndexTypeGetTypeID;
  static constexpr const char *pyClassName = "IndexType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

class PyFloatType : public PyConcreteType<PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat;
  static constexpr const char *pyClassName = "FloatType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float4E2M1FNType.
class PyFloat4E2M1FNType
    : public PyConcreteType<PyFloat4E2M1FNType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat4E2M1FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat4E2M1FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float4E2M1FNType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float6E2M3FNType.
class PyFloat6E2M3FNType
    : public PyConcreteType<PyFloat6E2M3FNType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat6E2M3FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat6E2M3FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float6E2M3FNType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float6E3M2FNType.
class PyFloat6E3M2FNType
    : public PyConcreteType<PyFloat6E3M2FNType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat6E3M2FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat6E3M2FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float6E3M2FNType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E4M3FNType.
class PyFloat8E4M3FNType
    : public PyConcreteType<PyFloat8E4M3FNType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E4M3FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E4M3FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3FNType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E5M2Type.
class PyFloat8E5M2Type : public PyConcreteType<PyFloat8E5M2Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E5M2;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E5M2TypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E5M2Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E4M3Type.
class PyFloat8E4M3Type : public PyConcreteType<PyFloat8E4M3Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E4M3;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E4M3TypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E4M3FNUZ.
class PyFloat8E4M3FNUZType
    : public PyConcreteType<PyFloat8E4M3FNUZType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E4M3FNUZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E4M3FNUZTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3FNUZType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E4M3B11FNUZ.
class PyFloat8E4M3B11FNUZType
    : public PyConcreteType<PyFloat8E4M3B11FNUZType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E4M3B11FNUZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E4M3B11FNUZTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3B11FNUZType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E5M2FNUZ.
class PyFloat8E5M2FNUZType
    : public PyConcreteType<PyFloat8E5M2FNUZType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E5M2FNUZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E5M2FNUZTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E5M2FNUZType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E3M4Type.
class PyFloat8E3M4Type : public PyConcreteType<PyFloat8E3M4Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E3M4;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E3M4TypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E3M4Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E8M0FNUType.
class PyFloat8E8M0FNUType
    : public PyConcreteType<PyFloat8E8M0FNUType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E8M0FNU;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E8M0FNUTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E8M0FNUType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - BF16Type.
class PyBF16Type : public PyConcreteType<PyBF16Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsABF16;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirBFloat16TypeGetTypeID;
  static constexpr const char *pyClassName = "BF16Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - F16Type.
class PyF16Type : public PyConcreteType<PyF16Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF16;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat16TypeGetTypeID;
  static constexpr const char *pyClassName = "F16Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - TF32Type.
class PyTF32Type : public PyConcreteType<PyTF32Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATF32;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloatTF32TypeGetTypeID;
  static constexpr const char *pyClassName = "FloatTF32Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - F32Type.
class PyF32Type : public PyConcreteType<PyF32Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF32;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat32TypeGetTypeID;
  static constexpr const char *pyClassName = "F32Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - F64Type.
class PyF64Type : public PyConcreteType<PyF64Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF64;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat64TypeGetTypeID;
  static constexpr const char *pyClassName = "F64Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// None Type subclass - NoneType.
class PyNoneType : public PyConcreteType<PyNoneType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsANone;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirNoneTypeGetTypeID;
  static constexpr const char *pyClassName = "NoneType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Complex Type subclass - ComplexType.
class PyComplexType : public PyConcreteType<PyComplexType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAComplex;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirComplexTypeGetTypeID;
  static constexpr const char *pyClassName = "ComplexType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Vector Type subclass - VectorType.
class PyVectorType : public PyConcreteType<PyVectorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAVector;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirVectorTypeGetTypeID;
  static constexpr const char *pyClassName = "VectorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);

private:
  static PyVectorType get(std::vector<int64_t> shape, PyType &elementType,
                          std::optional<nanobind::list> scalable,
                          std::optional<std::vector<int64_t>> scalableDims,
                          DefaultingPyLocation loc);
};

/// Ranked Tensor Type subclass - RankedTensorType.
class PyRankedTensorType
    : public PyConcreteType<PyRankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsARankedTensor;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirRankedTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "RankedTensorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Unranked Tensor Type subclass - UnrankedTensorType.
class PyUnrankedTensorType
    : public PyConcreteType<PyUnrankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUnrankedTensor;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirUnrankedTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "UnrankedTensorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Ranked MemRef Type subclass - MemRefType.
class PyMemRefType : public PyConcreteType<PyMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAMemRef;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirMemRefTypeGetTypeID;
  static constexpr const char *pyClassName = "MemRefType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Unranked MemRef Type subclass - UnrankedMemRefType.
class PyUnrankedMemRefType
    : public PyConcreteType<PyUnrankedMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUnrankedMemRef;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirUnrankedMemRefTypeGetTypeID;
  static constexpr const char *pyClassName = "UnrankedMemRefType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Tuple Type subclass - TupleType.
class PyTupleType : public PyConcreteType<PyTupleType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATuple;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTupleTypeGetTypeID;
  static constexpr const char *pyClassName = "TupleType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Function type.
class PyFunctionType : public PyConcreteType<PyFunctionType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFunction;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFunctionTypeGetTypeID;
  static constexpr const char *pyClassName = "FunctionType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Opaque Type subclass - OpaqueType.
class PyOpaqueType : public PyConcreteType<PyOpaqueType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAOpaque;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirOpaqueTypeGetTypeID;
  static constexpr const char *pyClassName = "OpaqueType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

} // namespace mlir::python

#endif // MLIR_BINDINGS_PYTHON_IRTYPES_H

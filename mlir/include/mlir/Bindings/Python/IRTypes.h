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
#include "mlir/Bindings/Python/IRCore.h"

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {

MLIR_PYTHON_API_EXPORTED int mlirTypeIsAIntegerOrFloat(MlirType type);

class MLIR_PYTHON_API_EXPORTED PyIntegerType
    : public PyConcreteType<PyIntegerType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAInteger;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirIntegerTypeGetTypeID;
  static constexpr const char *pyClassName = "IntegerType";
  static inline const MlirStringRef name = mlirIntegerTypeGetName();
  using PyConcreteType::PyConcreteType;

  enum Signedness { Signless, Signed, Unsigned };

  static void bindDerived(ClassTy &c);
};

/// Index Type subclass - IndexType.
class MLIR_PYTHON_API_EXPORTED PyIndexType
    : public PyConcreteType<PyIndexType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAIndex;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirIndexTypeGetTypeID;
  static constexpr const char *pyClassName = "IndexType";
  static inline const MlirStringRef name = mlirIndexTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

class MLIR_PYTHON_API_EXPORTED PyFloatType
    : public PyConcreteType<PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat;
  static constexpr const char *pyClassName = "FloatType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float4E2M1FNType.
class MLIR_PYTHON_API_EXPORTED PyFloat4E2M1FNType
    : public PyConcreteType<PyFloat4E2M1FNType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat4E2M1FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat4E2M1FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float4E2M1FNType";
  static inline const MlirStringRef name = mlirFloat4E2M1FNTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float6E2M3FNType.
class MLIR_PYTHON_API_EXPORTED PyFloat6E2M3FNType
    : public PyConcreteType<PyFloat6E2M3FNType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat6E2M3FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat6E2M3FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float6E2M3FNType";
  static inline const MlirStringRef name = mlirFloat6E2M3FNTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float6E3M2FNType.
class MLIR_PYTHON_API_EXPORTED PyFloat6E3M2FNType
    : public PyConcreteType<PyFloat6E3M2FNType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat6E3M2FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat6E3M2FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float6E3M2FNType";
  static inline const MlirStringRef name = mlirFloat6E3M2FNTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E4M3FNType.
class MLIR_PYTHON_API_EXPORTED PyFloat8E4M3FNType
    : public PyConcreteType<PyFloat8E4M3FNType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E4M3FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E4M3FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3FNType";
  static inline const MlirStringRef name = mlirFloat8E4M3FNTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E5M2Type.
class MLIR_PYTHON_API_EXPORTED PyFloat8E5M2Type
    : public PyConcreteType<PyFloat8E5M2Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E5M2;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E5M2TypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E5M2Type";
  static inline const MlirStringRef name = mlirFloat8E5M2TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E4M3Type.
class MLIR_PYTHON_API_EXPORTED PyFloat8E4M3Type
    : public PyConcreteType<PyFloat8E4M3Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E4M3;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E4M3TypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3Type";
  static inline const MlirStringRef name = mlirFloat8E4M3TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E4M3FNUZ.
class MLIR_PYTHON_API_EXPORTED PyFloat8E4M3FNUZType
    : public PyConcreteType<PyFloat8E4M3FNUZType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E4M3FNUZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E4M3FNUZTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3FNUZType";
  static inline const MlirStringRef name = mlirFloat8E4M3FNUZTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E4M3B11FNUZ.
class MLIR_PYTHON_API_EXPORTED PyFloat8E4M3B11FNUZType
    : public PyConcreteType<PyFloat8E4M3B11FNUZType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E4M3B11FNUZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E4M3B11FNUZTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3B11FNUZType";
  static inline const MlirStringRef name = mlirFloat8E4M3B11FNUZTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E5M2FNUZ.
class MLIR_PYTHON_API_EXPORTED PyFloat8E5M2FNUZType
    : public PyConcreteType<PyFloat8E5M2FNUZType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E5M2FNUZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E5M2FNUZTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E5M2FNUZType";
  static inline const MlirStringRef name = mlirFloat8E5M2FNUZTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E3M4Type.
class MLIR_PYTHON_API_EXPORTED PyFloat8E3M4Type
    : public PyConcreteType<PyFloat8E3M4Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E3M4;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E3M4TypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E3M4Type";
  static inline const MlirStringRef name = mlirFloat8E3M4TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - Float8E8M0FNUType.
class MLIR_PYTHON_API_EXPORTED PyFloat8E8M0FNUType
    : public PyConcreteType<PyFloat8E8M0FNUType, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E8M0FNU;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E8M0FNUTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E8M0FNUType";
  static inline const MlirStringRef name = mlirFloat8E8M0FNUTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - BF16Type.
class MLIR_PYTHON_API_EXPORTED PyBF16Type
    : public PyConcreteType<PyBF16Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsABF16;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirBFloat16TypeGetTypeID;
  static constexpr const char *pyClassName = "BF16Type";
  static inline const MlirStringRef name = mlirBF16TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - F16Type.
class MLIR_PYTHON_API_EXPORTED PyF16Type
    : public PyConcreteType<PyF16Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF16;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat16TypeGetTypeID;
  static constexpr const char *pyClassName = "F16Type";
  static inline const MlirStringRef name = mlirF16TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - TF32Type.
class MLIR_PYTHON_API_EXPORTED PyTF32Type
    : public PyConcreteType<PyTF32Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATF32;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloatTF32TypeGetTypeID;
  static constexpr const char *pyClassName = "FloatTF32Type";
  static inline const MlirStringRef name = mlirTF32TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - F32Type.
class MLIR_PYTHON_API_EXPORTED PyF32Type
    : public PyConcreteType<PyF32Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF32;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat32TypeGetTypeID;
  static constexpr const char *pyClassName = "F32Type";
  static inline const MlirStringRef name = mlirF32TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Floating Point Type subclass - F64Type.
class MLIR_PYTHON_API_EXPORTED PyF64Type
    : public PyConcreteType<PyF64Type, PyFloatType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF64;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat64TypeGetTypeID;
  static constexpr const char *pyClassName = "F64Type";
  static inline const MlirStringRef name = mlirF64TypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// None Type subclass - NoneType.
class MLIR_PYTHON_API_EXPORTED PyNoneType : public PyConcreteType<PyNoneType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsANone;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirNoneTypeGetTypeID;
  static constexpr const char *pyClassName = "NoneType";
  static inline const MlirStringRef name = mlirNoneTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Complex Type subclass - ComplexType.
class MLIR_PYTHON_API_EXPORTED PyComplexType
    : public PyConcreteType<PyComplexType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAComplex;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirComplexTypeGetTypeID;
  static constexpr const char *pyClassName = "ComplexType";
  static inline const MlirStringRef name = mlirComplexTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Shaped Type Interface - ShapedType
class MLIR_PYTHON_API_EXPORTED MLIR_PYTHON_API_EXPORTED PyShapedType
    : public PyConcreteType<PyShapedType> {
public:
  static const IsAFunctionTy isaFunction;
  static constexpr const char *pyClassName = "ShapedType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);

private:
  void requireHasRank();
};

/// Vector Type subclass - VectorType.
class MLIR_PYTHON_API_EXPORTED PyVectorType
    : public PyConcreteType<PyVectorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAVector;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirVectorTypeGetTypeID;
  static constexpr const char *pyClassName = "VectorType";
  static inline const MlirStringRef name = mlirVectorTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);

private:
  static PyVectorType
  getChecked(std::vector<int64_t> shape, PyType &elementType,
             std::optional<nanobind::list> scalable,
             std::optional<std::vector<int64_t>> scalableDims,
             DefaultingPyLocation loc);

  static PyVectorType get(std::vector<int64_t> shape, PyType &elementType,
                          std::optional<nanobind::list> scalable,
                          std::optional<std::vector<int64_t>> scalableDims,
                          DefaultingPyMlirContext context);
};

/// Ranked Tensor Type subclass - RankedTensorType.
class MLIR_PYTHON_API_EXPORTED PyRankedTensorType
    : public PyConcreteType<PyRankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsARankedTensor;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirRankedTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "RankedTensorType";
  static inline const MlirStringRef name = mlirRankedTensorTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Unranked Tensor Type subclass - UnrankedTensorType.
class MLIR_PYTHON_API_EXPORTED PyUnrankedTensorType
    : public PyConcreteType<PyUnrankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUnrankedTensor;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirUnrankedTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "UnrankedTensorType";
  static inline const MlirStringRef name = mlirUnrankedTensorTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Ranked MemRef Type subclass - MemRefType.
class MLIR_PYTHON_API_EXPORTED PyMemRefType
    : public PyConcreteType<PyMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAMemRef;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirMemRefTypeGetTypeID;
  static constexpr const char *pyClassName = "MemRefType";
  static inline const MlirStringRef name = mlirMemRefTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Unranked MemRef Type subclass - UnrankedMemRefType.
class MLIR_PYTHON_API_EXPORTED PyUnrankedMemRefType
    : public PyConcreteType<PyUnrankedMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUnrankedMemRef;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirUnrankedMemRefTypeGetTypeID;
  static constexpr const char *pyClassName = "UnrankedMemRefType";
  static inline const MlirStringRef name = mlirUnrankedMemRefTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Tuple Type subclass - TupleType.
class MLIR_PYTHON_API_EXPORTED PyTupleType
    : public PyConcreteType<PyTupleType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATuple;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTupleTypeGetTypeID;
  static constexpr const char *pyClassName = "TupleType";
  static inline const MlirStringRef name = mlirTupleTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Function type.
class MLIR_PYTHON_API_EXPORTED PyFunctionType
    : public PyConcreteType<PyFunctionType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFunction;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFunctionTypeGetTypeID;
  static constexpr const char *pyClassName = "FunctionType";
  static inline const MlirStringRef name = mlirFunctionTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

/// Opaque Type subclass - OpaqueType.
class MLIR_PYTHON_API_EXPORTED PyOpaqueType
    : public PyConcreteType<PyOpaqueType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAOpaque;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirOpaqueTypeGetTypeID;
  static constexpr const char *pyClassName = "OpaqueType";
  static inline const MlirStringRef name = mlirOpaqueTypeGetName();
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

MLIR_PYTHON_API_EXPORTED void populateIRTypes(nanobind::module_ &m);
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_IRTYPES_H

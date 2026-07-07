//===- DialectLLVM.cpp - Pybind module for LLVM dialect API support -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir-c/Target/LLVMIR.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

using namespace nanobind::literals;
using namespace mlir;
using namespace mlir::python::nanobind_adaptors;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace llvm {
//===--------------------------------------------------------------------===//
// StructType
//===--------------------------------------------------------------------===//

struct StructType : PyConcreteType<StructType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsALLVMStructType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirLLVMStructTypeGetTypeID;
  static constexpr const char *pyClassName = "StructType";
  static inline const MlirStringRef name = mlirLLVMStructTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_literal",
        [](const std::vector<PyType> &elements, bool packed,
           DefaultingPyLocation loc, DefaultingPyMlirContext context) {
          python::CollectDiagnosticsToStringScope scope(
              mlirLocationGetContext(loc));
          std::vector<MlirType> elements_(elements.size());
          std::copy(elements.begin(), elements.end(), elements_.begin());

          MlirType type = mlirLLVMStructTypeLiteralGetChecked(
              loc, elements.size(), elements_.data(), packed);
          if (mlirTypeIsNull(type)) {
            throw nb::value_error(scope.takeMessage().c_str());
          }
          return StructType(context->getRef(), type);
        },
        "elements"_a, nb::kw_only(), "packed"_a = false, "loc"_a = nb::none(),
        "context"_a = nb::none());

    c.def_static(
        "get_literal_unchecked",
        [](const std::vector<PyType> &elements, bool packed,
           DefaultingPyMlirContext context) {
          python::CollectDiagnosticsToStringScope scope(context.get()->get());

          std::vector<MlirType> elements_(elements.size());
          std::copy(elements.begin(), elements.end(), elements_.begin());

          MlirType type = mlirLLVMStructTypeLiteralGet(
              context.get()->get(), elements.size(), elements_.data(), packed);
          if (mlirTypeIsNull(type)) {
            throw nb::value_error(scope.takeMessage().c_str());
          }
          return StructType(context->getRef(), type);
        },
        "elements"_a, nb::kw_only(), "packed"_a = false,
        "context"_a = nb::none());

    c.def_static(
        "get_identified",
        [](const std::string &name, DefaultingPyMlirContext context) {
          return StructType(context->getRef(),
                            mlirLLVMStructTypeIdentifiedGet(
                                context.get()->get(),
                                mlirStringRefCreate(name.data(), name.size())));
        },
        "name"_a, nb::kw_only(), "context"_a = nb::none());

    c.def_static(
        "get_opaque",
        [](const std::string &name, DefaultingPyMlirContext context) {
          return StructType(context->getRef(),
                            mlirLLVMStructTypeOpaqueGet(
                                context.get()->get(),
                                mlirStringRefCreate(name.data(), name.size())));
        },
        "name"_a, "context"_a = nb::none());

    c.def(
        "set_body",
        [](const StructType &self, const std::vector<PyType> &elements,
           bool packed) {
          std::vector<MlirType> elements_(elements.size());
          std::copy(elements.begin(), elements.end(), elements_.begin());
          MlirLogicalResult result = mlirLLVMStructTypeSetBody(
              self, elements.size(), elements_.data(), packed);
          if (!mlirLogicalResultIsSuccess(result)) {
            throw nb::value_error(
                "Struct body already set to different content.");
          }
        },
        "elements"_a, nb::kw_only(), "packed"_a = false);

    c.def_static(
        "new_identified",
        [](const std::string &name, const std::vector<PyType> &elements,
           bool packed, DefaultingPyMlirContext context) {
          std::vector<MlirType> elements_(elements.size());
          std::copy(elements.begin(), elements.end(), elements_.begin());
          return StructType(context->getRef(),
                            mlirLLVMStructTypeIdentifiedNewGet(
                                context.get()->get(),
                                mlirStringRefCreate(name.data(), name.length()),
                                elements.size(), elements_.data(), packed));
        },
        "name"_a, "elements"_a, nb::kw_only(), "packed"_a = false,
        "context"_a = nb::none());

    c.def_prop_ro("name",
                  [](const StructType &type) -> std::optional<MlirStringRef> {
                    if (mlirLLVMStructTypeIsLiteral(type))
                      return std::nullopt;

                    return mlirLLVMStructTypeGetIdentifier(type);
                  });

    c.def_prop_ro("body", [](const StructType &type) -> nb::object {
      // Don't crash in absence of a body.
      if (mlirLLVMStructTypeIsOpaque(type))
        return nb::none();

      nb::list body;
      for (intptr_t i = 0, e = mlirLLVMStructTypeGetNumElementTypes(type);
           i < e; ++i) {
        body.append(mlirLLVMStructTypeGetElementType(type, i));
      }
      return body;
    });

    c.def_prop_ro("packed", [](const StructType &type) {
      return mlirLLVMStructTypeIsPacked(type);
    });

    c.def_prop_ro("opaque", [](const StructType &type) {
      return mlirLLVMStructTypeIsOpaque(type);
    });
  }
};

//===--------------------------------------------------------------------===//
// ArrayType
//===--------------------------------------------------------------------===//

struct ArrayType : PyConcreteType<ArrayType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsALLVMArrayType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirLLVMArrayTypeGetTypeID;
  static constexpr const char *pyClassName = "ArrayType";
  static inline const MlirStringRef name = mlirLLVMArrayTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elementType, unsigned numElements) {
          return ArrayType(elementType.getContext(),
                           mlirLLVMArrayTypeGet(elementType, numElements));
        },
        "element_type"_a, "num_elements"_a);
    c.def_prop_ro("element_type", [](const ArrayType &type) {
      return mlirLLVMArrayTypeGetElementType(type);
    });
    c.def_prop_ro("num_elements", [](const ArrayType &type) {
      return mlirLLVMArrayTypeGetNumElements(type);
    });
  }
};

//===--------------------------------------------------------------------===//
// PointerType
//===--------------------------------------------------------------------===//

struct PointerType : PyConcreteType<PointerType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsALLVMPointerType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirLLVMPointerTypeGetTypeID;
  static constexpr const char *pyClassName = "PointerType";
  static inline const MlirStringRef name = mlirLLVMPointerTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::optional<unsigned> addressSpace,
           DefaultingPyMlirContext context) {
          python::CollectDiagnosticsToStringScope scope(context.get()->get());
          MlirType type = mlirLLVMPointerTypeGet(
              context.get()->get(),
              addressSpace.has_value() ? *addressSpace : 0);
          if (mlirTypeIsNull(type)) {
            throw nb::value_error(scope.takeMessage().c_str());
          }
          return PointerType(context->getRef(), type);
        },
        "address_space"_a = nb::none(), nb::kw_only(),
        "context"_a = nb::none());
    c.def_prop_ro("address_space", [](const PointerType &type) {
      return mlirLLVMPointerTypeGetAddressSpace(type);
    });
  }
};

//===--------------------------------------------------------------------===//
// FunctionType
//===--------------------------------------------------------------------===//

struct FunctionType : PyConcreteType<FunctionType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsALLVMFunctionType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirLLVMFunctionTypeGetTypeID;
  static constexpr const char *pyClassName = "FunctionType";
  static inline const MlirStringRef name = mlirLLVMFunctionTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &resultType, const std::vector<PyType> &argumentTypes,
           bool isVarArg) {
          std::vector<MlirType> argTypes(argumentTypes.size());
          std::copy(argumentTypes.begin(), argumentTypes.end(),
                    argTypes.begin());
          return FunctionType(
              resultType.getContext(),
              mlirLLVMFunctionTypeGet(resultType, argTypes.size(),
                                      argTypes.data(), isVarArg));
        },
        "result_type"_a, "argument_types"_a, nb::kw_only(),
        "is_var_arg"_a = false);
    c.def_prop_ro("return_type", [](const FunctionType &type) {
      return mlirLLVMFunctionTypeGetReturnType(type);
    });
    c.def_prop_ro("num_inputs", [](const FunctionType &type) {
      return mlirLLVMFunctionTypeGetNumInputs(type);
    });
    c.def_prop_ro("inputs", [](const FunctionType &type) {
      nb::list inputs;
      for (intptr_t i = 0, e = mlirLLVMFunctionTypeGetNumInputs(type); i < e;
           ++i) {
        inputs.append(mlirLLVMFunctionTypeGetInput(type, i));
      }
      return inputs;
    });
    c.def_prop_ro("is_var_arg", [](const FunctionType &type) {
      return mlirLLVMFunctionTypeIsVarArg(type);
    });
  }
};

//===--------------------------------------------------------------------===//
// Metadata Attributes
//===--------------------------------------------------------------------===//

struct MDStringAttr : PyConcreteAttribute<MDStringAttr> {
  static constexpr IsAFunctionTy isaFunction = mlirLLVMAttrIsAMDStringAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirLLVMMDStringAttrGetTypeID;
  static constexpr const char *pyClassName = "MDStringAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &value, DefaultingPyMlirContext context) {
          return MDStringAttr(
              context->getRef(),
              mlirLLVMMDStringAttrGet(
                  context.get()->get(),
                  mlirStringRefCreate(value.data(), value.size())));
        },
        "value"_a, nb::kw_only(), "context"_a = nb::none());
    c.def_prop_ro("value", [](const MDStringAttr &self) {
      MlirStringRef ref = mlirLLVMMDStringAttrGetValue(self);
      return nb::str(ref.data, ref.length);
    });
  }
};

struct MDConstantAttr : PyConcreteAttribute<MDConstantAttr> {
  static constexpr IsAFunctionTy isaFunction = mlirLLVMAttrIsAMDConstantAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirLLVMMDConstantAttrGetTypeID;
  static constexpr const char *pyClassName = "MDConstantAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyAttribute &valueAttr, DefaultingPyMlirContext context) {
          return MDConstantAttr(
              context->getRef(),
              mlirLLVMMDConstantAttrGet(context.get()->get(), valueAttr));
        },
        "value"_a, nb::kw_only(), "context"_a = nb::none());
    c.def_prop_ro("value", [](const MDConstantAttr &self) {
      return mlirLLVMMDConstantAttrGetValue(self);
    });
  }
};

struct MDFuncAttr : PyConcreteAttribute<MDFuncAttr> {
  static constexpr IsAFunctionTy isaFunction = mlirLLVMAttrIsAMDFuncAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirLLVMMDFuncAttrGetTypeID;
  static constexpr const char *pyClassName = "MDFuncAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &name, DefaultingPyMlirContext context) {
          MlirAttribute symRef = mlirFlatSymbolRefAttrGet(
              context.get()->get(),
              mlirStringRefCreate(name.data(), name.size()));
          return MDFuncAttr(
              context->getRef(),
              mlirLLVMMDFuncAttrGet(context.get()->get(), symRef));
        },
        "name"_a, nb::kw_only(), "context"_a = nb::none());
    c.def_prop_ro("name", [](const MDFuncAttr &self) {
      MlirAttribute symRef = mlirLLVMMDFuncAttrGetName(self);
      MlirStringRef ref = mlirFlatSymbolRefAttrGetValue(symRef);
      return nb::str(ref.data, ref.length);
    });
  }
};

struct MDNodeAttr : PyConcreteAttribute<MDNodeAttr> {
  static constexpr IsAFunctionTy isaFunction = mlirLLVMAttrIsAMDNodeAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirLLVMMDNodeAttrGetTypeID;
  static constexpr const char *pyClassName = "MDNodeAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::vector<PyAttribute> &operands,
           DefaultingPyMlirContext context) {
          std::vector<MlirAttribute> operands_(operands.size());
          std::copy(operands.begin(), operands.end(), operands_.begin());
          return MDNodeAttr(context->getRef(),
                            mlirLLVMMDNodeAttrGet(context.get()->get(),
                                                  operands_.size(),
                                                  operands_.data()));
        },
        "operands"_a, nb::kw_only(), "context"_a = nb::none());
    c.def_prop_ro("num_operands", [](const MDNodeAttr &self) {
      return mlirLLVMMDNodeAttrGetNumOperands(self);
    });
    c.def("__getitem__", [](const MDNodeAttr &self, intptr_t index) {
      intptr_t n = mlirLLVMMDNodeAttrGetNumOperands(self);
      if (index < 0 || index >= n)
        throw nb::index_error("MDNodeAttr operand index out of range");
      return mlirLLVMMDNodeAttrGetOperand(self, index);
    });
    c.def("__len__", [](const MDNodeAttr &self) {
      return mlirLLVMMDNodeAttrGetNumOperands(self);
    });
  }
};

static void populateDialectLLVMSubmodule(nanobind::module_ &m) {
  StructType::bind(m);
  ArrayType::bind(m);
  PointerType::bind(m);
  FunctionType::bind(m);
  MDStringAttr::bind(m);
  MDConstantAttr::bind(m);
  MDFuncAttr::bind(m);
  MDNodeAttr::bind(m);

  m.def(
      "translate_module_to_llvmir",
      [](const PyOperation &module) {
        return mlirTranslateModuleToLLVMIRToString(module);
      },
      "module"_a, nb::rv_policy::take_ownership);
}
} // namespace llvm
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsLLVM, m) {
  m.doc() = "MLIR LLVM Dialect";

  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::llvm::populateDialectLLVMSubmodule(
      m);
}

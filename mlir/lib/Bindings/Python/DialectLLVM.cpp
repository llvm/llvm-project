//===- DialectLLVM.cpp - Pybind module for LLVM dialect API support -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

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
using namespace llvm;
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
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_literal",
        [](const std::vector<MlirType> &elements, bool packed, MlirLocation loc,
           DefaultingPyMlirContext context) {
          python::CollectDiagnosticsToStringScope scope(
              mlirLocationGetContext(loc));

          MlirType type = mlirLLVMStructTypeLiteralGetChecked(
              loc, elements.size(), elements.data(), packed);
          if (mlirTypeIsNull(type)) {
            throw nb::value_error(scope.takeMessage().c_str());
          }
          return StructType(context->getRef(), type);
        },
        "elements"_a, nb::kw_only(), "packed"_a = false, "loc"_a = nb::none(),
        "context"_a = nb::none());

    c.def_static(
        "get_literal_unchecked",
        [](const std::vector<MlirType> &elements, bool packed,
           DefaultingPyMlirContext context) {
          python::CollectDiagnosticsToStringScope scope(context.get()->get());

          MlirType type = mlirLLVMStructTypeLiteralGet(
              context.get()->get(), elements.size(), elements.data(), packed);
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
        [](MlirType self, const std::vector<MlirType> &elements, bool packed) {
          MlirLogicalResult result = mlirLLVMStructTypeSetBody(
              self, elements.size(), elements.data(), packed);
          if (!mlirLogicalResultIsSuccess(result)) {
            throw nb::value_error(
                "Struct body already set to different content.");
          }
        },
        "elements"_a, nb::kw_only(), "packed"_a = false);

    c.def_static(
        "new_identified",
        [](const std::string &name, const std::vector<MlirType> &elements,
           bool packed, DefaultingPyMlirContext context) {
          return StructType(context->getRef(),
                            mlirLLVMStructTypeIdentifiedNewGet(
                                context.get()->get(),
                                mlirStringRefCreate(name.data(), name.length()),
                                elements.size(), elements.data(), packed));
        },
        "name"_a, "elements"_a, nb::kw_only(), "packed"_a = false,
        "context"_a = nb::none());

    c.def_prop_ro("name", [](PyType type) -> std::optional<std::string> {
      if (mlirLLVMStructTypeIsLiteral(type))
        return std::nullopt;

      MlirStringRef stringRef = mlirLLVMStructTypeGetIdentifier(type);
      return StringRef(stringRef.data, stringRef.length).str();
    });

    c.def_prop_ro("body", [](PyType type) -> nb::object {
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

    c.def_prop_ro("packed",
                  [](PyType type) { return mlirLLVMStructTypeIsPacked(type); });

    c.def_prop_ro("opaque",
                  [](PyType type) { return mlirLLVMStructTypeIsOpaque(type); });
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
  using PyConcreteType::PyConcreteType;

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
    c.def_prop_ro("address_space", [](PyType type) {
      return mlirLLVMPointerTypeGetAddressSpace(type);
    });
  }
};

static void populateDialectLLVMSubmodule(nanobind::module_ &m) {
  StructType::bind(m);
  PointerType::bind(m);

  m.def(
      "translate_module_to_llvmir",
      [](MlirOperation module) {
        return mlirTranslateModuleToLLVMIRToString(module);
      },
      // clang-format off
      nb::sig("def translate_module_to_llvmir(module: " MAKE_MLIR_PYTHON_QUALNAME("ir.Operation") ") -> str"),
      // clang-format on
      "module"_a, nb::rv_policy::take_ownership);
}
} // namespace llvm
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsLLVM, m) {
  m.doc() = "MLIR LLVM Dialect";

  python::mlir::llvm::populateDialectLLVMSubmodule(m);
}

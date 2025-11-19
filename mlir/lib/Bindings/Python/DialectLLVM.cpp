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
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

using namespace nanobind::literals;

using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

static void populateDialectLLVMSubmodule(nanobind::module_ &m) {

  //===--------------------------------------------------------------------===//
  // StructType
  //===--------------------------------------------------------------------===//

  auto llvmStructType =
      mlir_type_subclass(m, "StructType", mlirTypeIsALLVMStructType);

  llvmStructType
      .def_classmethod(
          "get_literal",
          [](const nb::object &cls, const std::vector<MlirType> &elements,
             bool packed, MlirLocation loc) {
            CollectDiagnosticsToStringScope scope(mlirLocationGetContext(loc));

            MlirType type = mlirLLVMStructTypeLiteralGetChecked(
                loc, elements.size(), elements.data(), packed);
            if (mlirTypeIsNull(type)) {
              throw nb::value_error(scope.takeMessage().c_str());
            }
            return cls(type);
          },
          "cls"_a, "elements"_a, nb::kw_only(), "packed"_a = false,
          "loc"_a = nb::none())
      .def_classmethod(
          "get_literal_unchecked",
          [](const nb::object &cls, const std::vector<MlirType> &elements,
             bool packed, MlirContext context) {
            CollectDiagnosticsToStringScope scope(context);

            MlirType type = mlirLLVMStructTypeLiteralGet(
                context, elements.size(), elements.data(), packed);
            if (mlirTypeIsNull(type)) {
              throw nb::value_error(scope.takeMessage().c_str());
            }
            return cls(type);
          },
          "cls"_a, "elements"_a, nb::kw_only(), "packed"_a = false,
          "context"_a = nb::none());

  llvmStructType.def_classmethod(
      "get_identified",
      [](const nb::object &cls, const std::string &name, MlirContext context) {
        return cls(mlirLLVMStructTypeIdentifiedGet(
            context, mlirStringRefCreate(name.data(), name.size())));
      },
      "cls"_a, "name"_a, nb::kw_only(), "context"_a = nb::none());

  llvmStructType.def_classmethod(
      "get_opaque",
      [](const nb::object &cls, const std::string &name, MlirContext context) {
        return cls(mlirLLVMStructTypeOpaqueGet(
            context, mlirStringRefCreate(name.data(), name.size())));
      },
      "cls"_a, "name"_a, "context"_a = nb::none());

  llvmStructType.def(
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

  llvmStructType.def_classmethod(
      "new_identified",
      [](const nb::object &cls, const std::string &name,
         const std::vector<MlirType> &elements, bool packed, MlirContext ctx) {
        return cls(mlirLLVMStructTypeIdentifiedNewGet(
            ctx, mlirStringRefCreate(name.data(), name.length()),
            elements.size(), elements.data(), packed));
      },
      "cls"_a, "name"_a, "elements"_a, nb::kw_only(), "packed"_a = false,
      "context"_a = nb::none());

  llvmStructType.def_property_readonly(
      "name", [](MlirType type) -> std::optional<std::string> {
        if (mlirLLVMStructTypeIsLiteral(type))
          return std::nullopt;

        MlirStringRef stringRef = mlirLLVMStructTypeGetIdentifier(type);
        return StringRef(stringRef.data, stringRef.length).str();
      });

  llvmStructType.def_property_readonly("body", [](MlirType type) -> nb::object {
    // Don't crash in absence of a body.
    if (mlirLLVMStructTypeIsOpaque(type))
      return nb::none();

    nb::list body;
    for (intptr_t i = 0, e = mlirLLVMStructTypeGetNumElementTypes(type); i < e;
         ++i) {
      body.append(mlirLLVMStructTypeGetElementType(type, i));
    }
    return body;
  });

  llvmStructType.def_property_readonly(
      "packed", [](MlirType type) { return mlirLLVMStructTypeIsPacked(type); });

  llvmStructType.def_property_readonly(
      "opaque", [](MlirType type) { return mlirLLVMStructTypeIsOpaque(type); });

  //===--------------------------------------------------------------------===//
  // PointerType
  //===--------------------------------------------------------------------===//

  mlir_type_subclass(m, "PointerType", mlirTypeIsALLVMPointerType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, std::optional<unsigned> addressSpace,
             MlirContext context) {
            CollectDiagnosticsToStringScope scope(context);
            MlirType type = mlirLLVMPointerTypeGet(
                context, addressSpace.has_value() ? *addressSpace : 0);
            if (mlirTypeIsNull(type)) {
              throw nb::value_error(scope.takeMessage().c_str());
            }
            return cls(type);
          },
          "cls"_a, "address_space"_a = nb::none(), nb::kw_only(),
          "context"_a = nb::none())
      .def_property_readonly("address_space", [](MlirType type) {
        return mlirLLVMPointerTypeGetAddressSpace(type);
      });

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

NB_MODULE(_mlirDialectsLLVM, m) {
  m.doc() = "MLIR LLVM Dialect";

  populateDialectLLVMSubmodule(m);
}

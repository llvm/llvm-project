//===- IRModules.cpp - IR Submodules of pybind module ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRModules.h"
#include "PybindUtils.h"

#include "mlir-c/StandardTypes.h"

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;

//------------------------------------------------------------------------------
// Docstrings (trivial, non-duplicated docstrings are included inline).
//------------------------------------------------------------------------------

static const char kContextParseDocstring[] =
    R"(Parses a module's assembly format from a string.

Returns a new MlirModule or raises a ValueError if the parsing fails.

See also: https://mlir.llvm.org/docs/LangRef/
)";

static const char kContextParseType[] = R"(Parses the assembly form of a type.

Returns a Type object or raises a ValueError if the type cannot be parsed.

See also: https://mlir.llvm.org/docs/LangRef/#type-system
)";

static const char kOperationStrDunderDocstring[] =
    R"(Prints the assembly form of the operation with default options.

If more advanced control over the assembly formatting or I/O options is needed,
use the dedicated print method, which supports keyword arguments to customize
behavior.
)";

static const char kTypeStrDunderDocstring[] =
    R"(Prints the assembly form of the type.)";

static const char kDumpDocstring[] =
    R"(Dumps a debug representation of the object to stderr.)";

//------------------------------------------------------------------------------
// Conversion utilities.
//------------------------------------------------------------------------------

namespace {

/// Accumulates into a python string from a method that accepts an
/// MlirStringCallback.
struct PyPrintAccumulator {
  py::list parts;

  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](const char *part, intptr_t size, void *userData) {
      PyPrintAccumulator *printAccum =
          static_cast<PyPrintAccumulator *>(userData);
      py::str pyPart(part, size); // Decodes as UTF-8 by default.
      printAccum->parts.append(std::move(pyPart));
    };
  }

  py::str join() {
    py::str delim("", 0);
    return delim.attr("join")(parts);
  }
};

} // namespace

//------------------------------------------------------------------------------
// PyType.
//------------------------------------------------------------------------------

bool PyType::operator==(const PyType &other) {
  return mlirTypeEqual(type, other.type);
}

//------------------------------------------------------------------------------
// Standard type subclasses.
//------------------------------------------------------------------------------

namespace {

/// CRTP base classes for Python types that subclass Type and should be
/// castable from it (i.e. via something like IntegerType(t)).
template <typename T>
class PyConcreteType : public PyType {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  using ClassTy = py::class_<T, PyType>;
  using IsAFunctionTy = int (*)(MlirType);

  PyConcreteType() = default;
  PyConcreteType(MlirType t) : PyType(t) {}
  PyConcreteType(PyType &orig) : PyType(castFrom(orig)) {}

  static MlirType castFrom(PyType &orig) {
    if (!T::isaFunction(orig.type)) {
      auto origRepr = py::repr(py::cast(orig)).cast<std::string>();
      throw SetPyError(PyExc_ValueError, llvm::Twine("Cannot cast type to ") +
                                             T::pyClassName + " (from " +
                                             origRepr + ")");
    }
    return orig.type;
  }

  static void bind(py::module &m) {
    auto class_ = ClassTy(m, T::pyClassName);
    class_.def(py::init<PyType &>(), py::keep_alive<0, 1>());
    T::bindDerived(class_);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

class PyIntegerType : public PyConcreteType<PyIntegerType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAInteger;
  static constexpr const char *pyClassName = "IntegerType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "signless",
        [](PyMlirContext &context, unsigned width) {
          MlirType t = mlirIntegerTypeGet(context.context, width);
          return PyIntegerType(t);
        },
        py::keep_alive<0, 1>(), "Create a signless integer type");
    c.def_static(
        "signed",
        [](PyMlirContext &context, unsigned width) {
          MlirType t = mlirIntegerTypeSignedGet(context.context, width);
          return PyIntegerType(t);
        },
        py::keep_alive<0, 1>(), "Create a signed integer type");
    c.def_static(
        "unsigned",
        [](PyMlirContext &context, unsigned width) {
          MlirType t = mlirIntegerTypeUnsignedGet(context.context, width);
          return PyIntegerType(t);
        },
        py::keep_alive<0, 1>(), "Create an unsigned integer type");
    c.def_property_readonly(
        "width",
        [](PyIntegerType &self) { return mlirIntegerTypeGetWidth(self.type); },
        "Returns the width of the integer type");
    c.def_property_readonly(
        "is_signless",
        [](PyIntegerType &self) -> bool {
          return mlirIntegerTypeIsSignless(self.type);
        },
        "Returns whether this is a signless integer");
    c.def_property_readonly(
        "is_signed",
        [](PyIntegerType &self) -> bool {
          return mlirIntegerTypeIsSigned(self.type);
        },
        "Returns whether this is a signed integer");
    c.def_property_readonly(
        "is_unsigned",
        [](PyIntegerType &self) -> bool {
          return mlirIntegerTypeIsUnsigned(self.type);
        },
        "Returns whether this is an unsigned integer");
  }
};

} // namespace

//------------------------------------------------------------------------------
// Populates the pybind11 IR submodule.
//------------------------------------------------------------------------------

void mlir::python::populateIRSubmodule(py::module &m) {
  // Mapping of MlirContext
  py::class_<PyMlirContext>(m, "Context")
      .def(py::init<>())
      .def(
          "parse_module",
          [](PyMlirContext &self, const std::string module) {
            auto moduleRef =
                mlirModuleCreateParse(self.context, module.c_str());
            if (mlirModuleIsNull(moduleRef)) {
              throw SetPyError(
                  PyExc_ValueError,
                  "Unable to parse module assembly (see diagnostics)");
            }
            return PyModule(moduleRef);
          },
          py::keep_alive<0, 1>(), kContextParseDocstring)
      .def(
          "parse_type",
          [](PyMlirContext &self, std::string typeSpec) {
            MlirType type = mlirTypeParseGet(self.context, typeSpec.c_str());
            if (mlirTypeIsNull(type)) {
              throw SetPyError(PyExc_ValueError,
                               llvm::Twine("Unable to parse type: '") +
                                   typeSpec + "'");
            }
            return PyType(type);
          },
          py::keep_alive<0, 1>(), kContextParseType);

  // Mapping of Module
  py::class_<PyModule>(m, "Module")
      .def(
          "dump",
          [](PyModule &self) {
            mlirOperationDump(mlirModuleGetOperation(self.module));
          },
          kDumpDocstring)
      .def(
          "__str__",
          [](PyModule &self) {
            auto operation = mlirModuleGetOperation(self.module);
            PyPrintAccumulator printAccum;
            mlirOperationPrint(operation, printAccum.getCallback(),
                               printAccum.getUserData());
            return printAccum.join();
          },
          kOperationStrDunderDocstring);

  // Mapping of Type.
  py::class_<PyType>(m, "Type")
      .def("__eq__",
           [](PyType &self, py::object &other) {
             try {
               PyType otherType = other.cast<PyType>();
               return self == otherType;
             } catch (std::exception &e) {
               return false;
             }
           })
      .def(
          "dump", [](PyType &self) { mlirTypeDump(self.type); }, kDumpDocstring)
      .def(
          "__str__",
          [](PyType &self) {
            PyPrintAccumulator printAccum;
            mlirTypePrint(self.type, printAccum.getCallback(),
                          printAccum.getUserData());
            return printAccum.join();
          },
          kTypeStrDunderDocstring)
      .def("__repr__", [](PyType &self) {
        // Generally, assembly formats are not printed for __repr__ because
        // this can cause exceptionally long debug output and exceptions.
        // However, types are an exception as they typically have compact
        // assembly forms and printing them is useful.
        PyPrintAccumulator printAccum;
        printAccum.parts.append("Type(");
        mlirTypePrint(self.type, printAccum.getCallback(),
                      printAccum.getUserData());
        printAccum.parts.append(")");
        return printAccum.join();
      });

  // Standard type bindings.
  PyIntegerType::bind(m);
}

//===- IRModules.cpp - IR Submodules of pybind module ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRModules.h"
#include "PybindUtils.h"

#include "mlir-c/StandardAttributes.h"
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

static const char kContextParseTypeDocstring[] =
    R"(Parses the assembly form of a type.

Returns a Type object or raises a ValueError if the type cannot be parsed.

See also: https://mlir.llvm.org/docs/LangRef/#type-system
)";

static const char kContextGetUnknownLocationDocstring[] =
    R"(Gets a Location representing an unknown location)";

static const char kContextGetFileLocationDocstring[] =
    R"(Gets a Location representing a file, line and column)";

static const char kContextCreateBlockDocstring[] =
    R"(Creates a detached block)";

static const char kContextCreateRegionDocstring[] =
    R"(Creates a detached region)";

static const char kRegionAppendBlockDocstring[] =
    R"(Appends a block to a region.

Raises:
  ValueError: If the block is already attached to another region.
)";

static const char kRegionInsertBlockDocstring[] =
    R"(Inserts a block at a postiion in a region.

Raises:
  ValueError: If the block is already attached to another region.
)";

static const char kRegionFirstBlockDocstring[] =
    R"(Gets the first block in a region.

Blocks can also be accessed via the `blocks` container.

Raises:
  IndexError: If the region has no blocks.
)";

static const char kBlockNextInRegionDocstring[] =
    R"(Gets the next block in the enclosing region.

Blocks can also be accessed via the `blocks` container of the owning region.
This method exists to mirror the lower level API and should not be preferred.

Raises:
  IndexError: If there are no further blocks.
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

/// Accumulates into a python string from a method that is expected to make
/// one (no more, no less) call to the callback (asserts internally on
/// violation).
struct PySinglePartStringAccumulator {
  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](const char *part, intptr_t size, void *userData) {
      PySinglePartStringAccumulator *accum =
          static_cast<PySinglePartStringAccumulator *>(userData);
      assert(!accum->invoked &&
             "PySinglePartStringAccumulator called back multiple times");
      accum->invoked = true;
      accum->value = py::str(part, size);
    };
  }

  py::str takeValue() {
    assert(invoked && "PySinglePartStringAccumulator not called back");
    return std::move(value);
  }

private:
  py::str value;
  bool invoked = false;
};

} // namespace

//------------------------------------------------------------------------------
// PyBlock, PyRegion, and PyOperation.
//------------------------------------------------------------------------------

void PyRegion::attachToParent() {
  if (!detached) {
    throw SetPyError(PyExc_ValueError, "Region is already attached to an op");
  }
  detached = false;
}

void PyBlock::attachToParent() {
  if (!detached) {
    throw SetPyError(PyExc_ValueError, "Block is already attached to an op");
  }
  detached = false;
}

//------------------------------------------------------------------------------
// PyAttribute.
//------------------------------------------------------------------------------

bool PyAttribute::operator==(const PyAttribute &other) {
  return mlirAttributeEqual(attr, other.attr);
}

//------------------------------------------------------------------------------
// PyNamedAttribute.
//------------------------------------------------------------------------------

PyNamedAttribute::PyNamedAttribute(MlirAttribute attr, std::string ownedName)
    : ownedName(new std::string(std::move(ownedName))) {
  namedAttr = mlirNamedAttributeGet(this->ownedName->c_str(), attr);
}

//------------------------------------------------------------------------------
// PyType.
//------------------------------------------------------------------------------

bool PyType::operator==(const PyType &other) {
  return mlirTypeEqual(type, other.type);
}

//------------------------------------------------------------------------------
// Standard attribute subclasses.
//------------------------------------------------------------------------------

namespace {

/// CRTP base classes for Python attributes that subclass Attribute and should
/// be castable from it (i.e. via something like StringAttr(attr)).
template <typename T>
class PyConcreteAttribute : public PyAttribute {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  using ClassTy = py::class_<T, PyAttribute>;
  using IsAFunctionTy = int (*)(MlirAttribute);

  PyConcreteAttribute() = default;
  PyConcreteAttribute(MlirAttribute attr) : PyAttribute(attr) {}
  PyConcreteAttribute(PyAttribute &orig)
      : PyConcreteAttribute(castFrom(orig)) {}

  static MlirAttribute castFrom(PyAttribute &orig) {
    if (!T::isaFunction(orig.attr)) {
      auto origRepr = py::repr(py::cast(orig)).cast<std::string>();
      throw SetPyError(PyExc_ValueError,
                       llvm::Twine("Cannot cast attribute to ") +
                           T::pyClassName + " (from " + origRepr + ")");
    }
    return orig.attr;
  }

  static void bind(py::module &m) {
    auto cls = ClassTy(m, T::pyClassName);
    cls.def(py::init<PyAttribute &>(), py::keep_alive<0, 1>());
    T::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

class PyStringAttribute : public PyConcreteAttribute<PyStringAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAString;
  static constexpr const char *pyClassName = "StringAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyMlirContext &context, std::string value) {
          MlirAttribute attr =
              mlirStringAttrGet(context.context, value.size(), &value[0]);
          return PyStringAttribute(attr);
        },
        py::keep_alive<0, 1>(), "Gets a uniqued string attribute");
    c.def_static(
        "get_typed",
        [](PyType &type, std::string value) {
          MlirAttribute attr =
              mlirStringAttrTypedGet(type.type, value.size(), &value[0]);
          return PyStringAttribute(attr);
        },
        py::keep_alive<0, 1>(),
        "Gets a uniqued string attribute associated to a type");
    c.def_property_readonly(
        "value",
        [](PyStringAttribute &self) {
          PySinglePartStringAccumulator accum;
          mlirStringAttrGetValue(self.attr, accum.getCallback(),
                                 accum.getUserData());
          return accum.takeValue();
        },
        "Returns the value of the string attribute");
  }
};

} // namespace

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
    auto cls = ClassTy(m, T::pyClassName);
    cls.def(py::init<PyType &>(), py::keep_alive<0, 1>());
    T::bindDerived(cls);
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
        "get_signless",
        [](PyMlirContext &context, unsigned width) {
          MlirType t = mlirIntegerTypeGet(context.context, width);
          return PyIntegerType(t);
        },
        py::keep_alive<0, 1>(), "Create a signless integer type");
    c.def_static(
        "get_signed",
        [](PyMlirContext &context, unsigned width) {
          MlirType t = mlirIntegerTypeSignedGet(context.context, width);
          return PyIntegerType(t);
        },
        py::keep_alive<0, 1>(), "Create a signed integer type");
    c.def_static(
        "get_unsigned",
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

/// Index Type subclass - IndexType.
class PyIndexType : public PyConcreteType<PyIndexType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAIndex;
  static constexpr const char *pyClassName = "IndexType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def(py::init([](PyMlirContext &context) {
            MlirType t = mlirIndexTypeGet(context.context);
            return PyIndexType(t);
          }),
          py::keep_alive<0, 1>(), "Create a index type.");
  }
};

/// Floating Point Type subclass - BF16Type.
class PyBF16Type : public PyConcreteType<PyBF16Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsABF16;
  static constexpr const char *pyClassName = "BF16Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def(py::init([](PyMlirContext &context) {
            MlirType t = mlirBF16TypeGet(context.context);
            return PyBF16Type(t);
          }),
          py::keep_alive<0, 1>(), "Create a bf16 type.");
  }
};

/// Floating Point Type subclass - F16Type.
class PyF16Type : public PyConcreteType<PyF16Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF16;
  static constexpr const char *pyClassName = "F16Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def(py::init([](PyMlirContext &context) {
            MlirType t = mlirF16TypeGet(context.context);
            return PyF16Type(t);
          }),
          py::keep_alive<0, 1>(), "Create a f16 type.");
  }
};

/// Floating Point Type subclass - F32Type.
class PyF32Type : public PyConcreteType<PyF32Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF32;
  static constexpr const char *pyClassName = "F32Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def(py::init([](PyMlirContext &context) {
            MlirType t = mlirF32TypeGet(context.context);
            return PyF32Type(t);
          }),
          py::keep_alive<0, 1>(), "Create a f32 type.");
  }
};

/// Floating Point Type subclass - F64Type.
class PyF64Type : public PyConcreteType<PyF64Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF64;
  static constexpr const char *pyClassName = "F64Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def(py::init([](PyMlirContext &context) {
            MlirType t = mlirF64TypeGet(context.context);
            return PyF64Type(t);
          }),
          py::keep_alive<0, 1>(), "Create a f64 type.");
  }
};

/// None Type subclass - NoneType.
class PyNoneType : public PyConcreteType<PyNoneType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsANone;
  static constexpr const char *pyClassName = "NoneType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def(py::init([](PyMlirContext &context) {
            MlirType t = mlirNoneTypeGet(context.context);
            return PyNoneType(t);
          }),
          py::keep_alive<0, 1>(), "Create a none type.");
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
            // TODO: Rework error reporting once diagnostic engine is exposed
            // in C API.
            if (mlirModuleIsNull(moduleRef)) {
              throw SetPyError(
                  PyExc_ValueError,
                  "Unable to parse module assembly (see diagnostics)");
            }
            return PyModule(moduleRef);
          },
          py::keep_alive<0, 1>(), kContextParseDocstring)
      .def(
          "parse_attr",
          [](PyMlirContext &self, std::string attrSpec) {
            MlirAttribute type =
                mlirAttributeParseGet(self.context, attrSpec.c_str());
            // TODO: Rework error reporting once diagnostic engine is exposed
            // in C API.
            if (mlirAttributeIsNull(type)) {
              throw SetPyError(PyExc_ValueError,
                               llvm::Twine("Unable to parse attribute: '") +
                                   attrSpec + "'");
            }
            return PyAttribute(type);
          },
          py::keep_alive<0, 1>())
      .def(
          "parse_type",
          [](PyMlirContext &self, std::string typeSpec) {
            MlirType type = mlirTypeParseGet(self.context, typeSpec.c_str());
            // TODO: Rework error reporting once diagnostic engine is exposed
            // in C API.
            if (mlirTypeIsNull(type)) {
              throw SetPyError(PyExc_ValueError,
                               llvm::Twine("Unable to parse type: '") +
                                   typeSpec + "'");
            }
            return PyType(type);
          },
          py::keep_alive<0, 1>(), kContextParseTypeDocstring)
      .def(
          "get_unknown_location",
          [](PyMlirContext &self) {
            return PyLocation(mlirLocationUnknownGet(self.context));
          },
          py::keep_alive<0, 1>(), kContextGetUnknownLocationDocstring)
      .def(
          "get_file_location",
          [](PyMlirContext &self, std::string filename, int line, int col) {
            return PyLocation(mlirLocationFileLineColGet(
                self.context, filename.c_str(), line, col));
          },
          py::keep_alive<0, 1>(), kContextGetFileLocationDocstring,
          py::arg("filename"), py::arg("line"), py::arg("col"))
      .def(
          "create_region",
          [](PyMlirContext &self) {
            // The creating context is explicitly captured on regions to
            // facilitate illegal assemblies of objects from multiple contexts
            // that would invalidate the memory model.
            return PyRegion(self.context, mlirRegionCreate(),
                            /*detached=*/true);
          },
          py::keep_alive<0, 1>(), kContextCreateRegionDocstring)
      .def(
          "create_block",
          [](PyMlirContext &self, std::vector<PyType> pyTypes) {
            // In order for the keep_alive extend the proper lifetime, all
            // types must be from the same context.
            for (auto pyType : pyTypes) {
              if (!mlirContextEqual(mlirTypeGetContext(pyType.type),
                                    self.context)) {
                throw SetPyError(
                    PyExc_ValueError,
                    "All types used to construct a block must be from "
                    "the same context as the block");
              }
            }
            llvm::SmallVector<MlirType, 4> types(pyTypes.begin(),
                                                 pyTypes.end());
            return PyBlock(self.context,
                           mlirBlockCreate(types.size(), &types[0]),
                           /*detached=*/true);
          },
          py::keep_alive<0, 1>(), kContextCreateBlockDocstring);

  py::class_<PyLocation>(m, "Location").def("__repr__", [](PyLocation &self) {
    PyPrintAccumulator printAccum;
    mlirLocationPrint(self.loc, printAccum.getCallback(),
                      printAccum.getUserData());
    return printAccum.join();
  });

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

  // Mapping of PyRegion.
  py::class_<PyRegion>(m, "Region")
      .def(
          "append_block",
          [](PyRegion &self, PyBlock &block) {
            if (!mlirContextEqual(self.context, block.context)) {
              throw SetPyError(
                  PyExc_ValueError,
                  "Block must have been created from the same context as "
                  "this region");
            }

            block.attachToParent();
            mlirRegionAppendOwnedBlock(self.region, block.block);
          },
          kRegionAppendBlockDocstring)
      .def(
          "insert_block",
          [](PyRegion &self, int pos, PyBlock &block) {
            if (!mlirContextEqual(self.context, block.context)) {
              throw SetPyError(
                  PyExc_ValueError,
                  "Block must have been created from the same context as "
                  "this region");
            }
            block.attachToParent();
            // TODO: Make this return a failure and raise if out of bounds.
            mlirRegionInsertOwnedBlock(self.region, pos, block.block);
          },
          kRegionInsertBlockDocstring)
      .def_property_readonly(
          "first_block",
          [](PyRegion &self) {
            MlirBlock block = mlirRegionGetFirstBlock(self.region);
            if (mlirBlockIsNull(block)) {
              throw SetPyError(PyExc_IndexError, "Region has no blocks");
            }
            return PyBlock(self.context, block, /*detached=*/false);
          },
          kRegionFirstBlockDocstring);

  // Mapping of PyBlock.
  py::class_<PyBlock>(m, "Block")
      .def_property_readonly(
          "next_in_region",
          [](PyBlock &self) {
            MlirBlock block = mlirBlockGetNextInRegion(self.block);
            if (mlirBlockIsNull(block)) {
              throw SetPyError(PyExc_IndexError,
                               "Attempt to read past last block");
            }
            return PyBlock(self.context, block, /*detached=*/false);
          },
          py::keep_alive<0, 1>(), kBlockNextInRegionDocstring)
      .def(
          "__str__",
          [](PyBlock &self) {
            PyPrintAccumulator printAccum;
            mlirBlockPrint(self.block, printAccum.getCallback(),
                           printAccum.getUserData());
            return printAccum.join();
          },
          kTypeStrDunderDocstring);

  // Mapping of Type.
  py::class_<PyAttribute>(m, "Attribute")
      .def(
          "get_named",
          [](PyAttribute &self, std::string name) {
            return PyNamedAttribute(self.attr, std::move(name));
          },
          py::keep_alive<0, 1>(), "Binds a name to the attribute")
      .def("__eq__",
           [](PyAttribute &self, py::object &other) {
             try {
               PyAttribute otherAttribute = other.cast<PyAttribute>();
               return self == otherAttribute;
             } catch (std::exception &e) {
               return false;
             }
           })
      .def(
          "dump", [](PyAttribute &self) { mlirAttributeDump(self.attr); },
          kDumpDocstring)
      .def(
          "__str__",
          [](PyAttribute &self) {
            PyPrintAccumulator printAccum;
            mlirAttributePrint(self.attr, printAccum.getCallback(),
                               printAccum.getUserData());
            return printAccum.join();
          },
          kTypeStrDunderDocstring)
      .def("__repr__", [](PyAttribute &self) {
        // Generally, assembly formats are not printed for __repr__ because
        // this can cause exceptionally long debug output and exceptions.
        // However, attribute values are generally considered useful and are
        // printed. This may need to be re-evaluated if debug dumps end up
        // being excessive.
        PyPrintAccumulator printAccum;
        printAccum.parts.append("Attribute(");
        mlirAttributePrint(self.attr, printAccum.getCallback(),
                           printAccum.getUserData());
        printAccum.parts.append(")");
        return printAccum.join();
      });

  py::class_<PyNamedAttribute>(m, "NamedAttribute")
      .def("__repr__",
           [](PyNamedAttribute &self) {
             PyPrintAccumulator printAccum;
             printAccum.parts.append("NamedAttribute(");
             printAccum.parts.append(self.namedAttr.name);
             printAccum.parts.append("=");
             mlirAttributePrint(self.namedAttr.attribute,
                                printAccum.getCallback(),
                                printAccum.getUserData());
             printAccum.parts.append(")");
             return printAccum.join();
           })
      .def_property_readonly(
          "name",
          [](PyNamedAttribute &self) {
            return py::str(self.namedAttr.name, strlen(self.namedAttr.name));
          },
          "The name of the NamedAttribute binding")
      .def_property_readonly(
          "attr",
          [](PyNamedAttribute &self) {
            return PyAttribute(self.namedAttr.attribute);
          },
          py::keep_alive<0, 1>(),
          "The underlying generic attribute of the NamedAttribute binding");

  // Standard attribute bindings.
  PyStringAttribute::bind(m);

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
  PyIndexType::bind(m);
  PyBF16Type::bind(m);
  PyF16Type::bind(m);
  PyF32Type::bind(m);
  PyF64Type::bind(m);
  PyNoneType::bind(m);
}

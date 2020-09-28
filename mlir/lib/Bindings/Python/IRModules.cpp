//===- IRModules.cpp - IR Submodules of pybind module ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRModules.h"
#include "PybindUtils.h"

#include "mlir-c/Registration.h"
#include "mlir-c/StandardAttributes.h"
#include "mlir-c/StandardTypes.h"
#include "llvm/ADT/SmallVector.h"
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;

using llvm::SmallVector;

//------------------------------------------------------------------------------
// Docstrings (trivial, non-duplicated docstrings are included inline).
//------------------------------------------------------------------------------

static const char kContextCreateOperationDocstring[] =
    R"(Creates a new operation.

Args:
  name: Operation name (e.g. "dialect.operation").
  location: A Location object.
  results: Sequence of Type representing op result types.
  attributes: Dict of str:Attribute.
  successors: List of Block for the operation's successors.
  regions: Number of regions to create.

Returns:
  A new "detached" Operation object. Detached operations can be added
  to blocks, which causes them to become "attached."
)";

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

static const char kAppendBlockDocstring[] =
    R"(Appends a new block, with argument types as positional args.

Returns:
  The created block.
)";

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
// Type-checking utilities.
//------------------------------------------------------------------------------

namespace {

/// Checks whether the given type is an integer or float type.
int mlirTypeIsAIntegerOrFloat(MlirType type) {
  return mlirTypeIsAInteger(type) || mlirTypeIsABF16(type) ||
         mlirTypeIsAF16(type) || mlirTypeIsAF32(type) || mlirTypeIsAF64(type);
}

} // namespace

//------------------------------------------------------------------------------
// Collections.
//------------------------------------------------------------------------------

namespace {

class PyRegionIterator {
public:
  PyRegionIterator(PyOperationRef operation)
      : operation(std::move(operation)) {}

  PyRegionIterator &dunderIter() { return *this; }

  PyRegion dunderNext() {
    operation->checkValid();
    if (nextIndex >= mlirOperationGetNumRegions(operation->get())) {
      throw py::stop_iteration();
    }
    MlirRegion region = mlirOperationGetRegion(operation->get(), nextIndex++);
    return PyRegion(operation, region);
  }

  static void bind(py::module &m) {
    py::class_<PyRegionIterator>(m, "RegionIterator")
        .def("__iter__", &PyRegionIterator::dunderIter)
        .def("__next__", &PyRegionIterator::dunderNext);
  }

private:
  PyOperationRef operation;
  int nextIndex = 0;
};

/// Regions of an op are fixed length and indexed numerically so are represented
/// with a sequence-like container.
class PyRegionList {
public:
  PyRegionList(PyOperationRef operation) : operation(std::move(operation)) {}

  intptr_t dunderLen() {
    operation->checkValid();
    return mlirOperationGetNumRegions(operation->get());
  }

  PyRegion dunderGetItem(intptr_t index) {
    // dunderLen checks validity.
    if (index < 0 || index >= dunderLen()) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to access out of bounds region");
    }
    MlirRegion region = mlirOperationGetRegion(operation->get(), index);
    return PyRegion(operation, region);
  }

  static void bind(py::module &m) {
    py::class_<PyRegionList>(m, "ReqionSequence")
        .def("__len__", &PyRegionList::dunderLen)
        .def("__getitem__", &PyRegionList::dunderGetItem);
  }

private:
  PyOperationRef operation;
};

class PyBlockIterator {
public:
  PyBlockIterator(PyOperationRef operation, MlirBlock next)
      : operation(std::move(operation)), next(next) {}

  PyBlockIterator &dunderIter() { return *this; }

  PyBlock dunderNext() {
    operation->checkValid();
    if (mlirBlockIsNull(next)) {
      throw py::stop_iteration();
    }

    PyBlock returnBlock(operation, next);
    next = mlirBlockGetNextInRegion(next);
    return returnBlock;
  }

  static void bind(py::module &m) {
    py::class_<PyBlockIterator>(m, "BlockIterator")
        .def("__iter__", &PyBlockIterator::dunderIter)
        .def("__next__", &PyBlockIterator::dunderNext);
  }

private:
  PyOperationRef operation;
  MlirBlock next;
};

/// Blocks are exposed by the C-API as a forward-only linked list. In Python,
/// we present them as a more full-featured list-like container but optimzie
/// it for forward iteration. Blocks are always owned by a region.
class PyBlockList {
public:
  PyBlockList(PyOperationRef operation, MlirRegion region)
      : operation(std::move(operation)), region(region) {}

  PyBlockIterator dunderIter() {
    operation->checkValid();
    return PyBlockIterator(operation, mlirRegionGetFirstBlock(region));
  }

  intptr_t dunderLen() {
    operation->checkValid();
    intptr_t count = 0;
    MlirBlock block = mlirRegionGetFirstBlock(region);
    while (!mlirBlockIsNull(block)) {
      count += 1;
      block = mlirBlockGetNextInRegion(block);
    }
    return count;
  }

  PyBlock dunderGetItem(intptr_t index) {
    operation->checkValid();
    if (index < 0) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to access out of bounds block");
    }
    MlirBlock block = mlirRegionGetFirstBlock(region);
    while (!mlirBlockIsNull(block)) {
      if (index == 0) {
        return PyBlock(operation, block);
      }
      block = mlirBlockGetNextInRegion(block);
      index -= 1;
    }
    throw SetPyError(PyExc_IndexError, "attempt to access out of bounds block");
  }

  PyBlock appendBlock(py::args pyArgTypes) {
    operation->checkValid();
    llvm::SmallVector<MlirType, 4> argTypes;
    argTypes.reserve(pyArgTypes.size());
    for (auto &pyArg : pyArgTypes) {
      argTypes.push_back(pyArg.cast<PyType &>().type);
    }

    MlirBlock block = mlirBlockCreate(argTypes.size(), argTypes.data());
    mlirRegionAppendOwnedBlock(region, block);
    return PyBlock(operation, block);
  }

  static void bind(py::module &m) {
    py::class_<PyBlockList>(m, "BlockList")
        .def("__getitem__", &PyBlockList::dunderGetItem)
        .def("__iter__", &PyBlockList::dunderIter)
        .def("__len__", &PyBlockList::dunderLen)
        .def("append", &PyBlockList::appendBlock, kAppendBlockDocstring);
  }

private:
  PyOperationRef operation;
  MlirRegion region;
};

class PyOperationIterator {
public:
  PyOperationIterator(PyOperationRef parentOperation, MlirOperation next)
      : parentOperation(std::move(parentOperation)), next(next) {}

  PyOperationIterator &dunderIter() { return *this; }

  py::object dunderNext() {
    parentOperation->checkValid();
    if (mlirOperationIsNull(next)) {
      throw py::stop_iteration();
    }

    PyOperationRef returnOperation =
        PyOperation::forOperation(parentOperation->getContext(), next);
    next = mlirOperationGetNextInBlock(next);
    return returnOperation.releaseObject();
  }

  static void bind(py::module &m) {
    py::class_<PyOperationIterator>(m, "OperationIterator")
        .def("__iter__", &PyOperationIterator::dunderIter)
        .def("__next__", &PyOperationIterator::dunderNext);
  }

private:
  PyOperationRef parentOperation;
  MlirOperation next;
};

/// Operations are exposed by the C-API as a forward-only linked list. In
/// Python, we present them as a more full-featured list-like container but
/// optimzie it for forward iteration. Iterable operations are always owned
/// by a block.
class PyOperationList {
public:
  PyOperationList(PyOperationRef parentOperation, MlirBlock block)
      : parentOperation(std::move(parentOperation)), block(block) {}

  PyOperationIterator dunderIter() {
    parentOperation->checkValid();
    return PyOperationIterator(parentOperation,
                               mlirBlockGetFirstOperation(block));
  }

  intptr_t dunderLen() {
    parentOperation->checkValid();
    intptr_t count = 0;
    MlirOperation childOp = mlirBlockGetFirstOperation(block);
    while (!mlirOperationIsNull(childOp)) {
      count += 1;
      childOp = mlirOperationGetNextInBlock(childOp);
    }
    return count;
  }

  py::object dunderGetItem(intptr_t index) {
    parentOperation->checkValid();
    if (index < 0) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to access out of bounds operation");
    }
    MlirOperation childOp = mlirBlockGetFirstOperation(block);
    while (!mlirOperationIsNull(childOp)) {
      if (index == 0) {
        return PyOperation::forOperation(parentOperation->getContext(), childOp)
            .releaseObject();
      }
      childOp = mlirOperationGetNextInBlock(childOp);
      index -= 1;
    }
    throw SetPyError(PyExc_IndexError,
                     "attempt to access out of bounds operation");
  }

  void insert(int index, PyOperation &newOperation) {
    parentOperation->checkValid();
    newOperation.checkValid();
    if (index < 0) {
      throw SetPyError(
          PyExc_IndexError,
          "only positive insertion indices are supported for operations");
    }
    if (newOperation.isAttached()) {
      throw SetPyError(
          PyExc_ValueError,
          "attempt to insert an operation that has already been inserted");
    }
    // TODO: Needing to do this check is unfortunate, especially since it will
    // be a forward-scan, just like the following call to
    // mlirBlockInsertOwnedOperation. Switch to insert before/after once
    // D88148 lands.
    if (index > dunderLen()) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to insert operation past end");
    }
    mlirBlockInsertOwnedOperation(block, index, newOperation.get());
    newOperation.setAttached();
    // TODO: Rework the parentKeepAlive so as to avoid ownership hazards under
    // the new ownership.
  }

  static void bind(py::module &m) {
    py::class_<PyOperationList>(m, "OperationList")
        .def("__getitem__", &PyOperationList::dunderGetItem)
        .def("__iter__", &PyOperationList::dunderIter)
        .def("__len__", &PyOperationList::dunderLen)
        .def("insert", &PyOperationList::insert, py::arg("index"),
             py::arg("operation"),
             "Inserts an operation at an indexed position");
  }

private:
  PyOperationRef parentOperation;
  MlirBlock block;
};

} // namespace

//------------------------------------------------------------------------------
// PyMlirContext
//------------------------------------------------------------------------------

PyMlirContext::PyMlirContext(MlirContext context) : context(context) {
  py::gil_scoped_acquire acquire;
  auto &liveContexts = getLiveContexts();
  liveContexts[context.ptr] = this;
}

PyMlirContext::~PyMlirContext() {
  // Note that the only public way to construct an instance is via the
  // forContext method, which always puts the associated handle into
  // liveContexts.
  py::gil_scoped_acquire acquire;
  getLiveContexts().erase(context.ptr);
  mlirContextDestroy(context);
}

PyMlirContext *PyMlirContext::createNewContextForInit() {
  MlirContext context = mlirContextCreate();
  mlirRegisterAllDialects(context);
  return new PyMlirContext(context);
}

PyMlirContextRef PyMlirContext::forContext(MlirContext context) {
  py::gil_scoped_acquire acquire;
  auto &liveContexts = getLiveContexts();
  auto it = liveContexts.find(context.ptr);
  if (it == liveContexts.end()) {
    // Create.
    PyMlirContext *unownedContextWrapper = new PyMlirContext(context);
    py::object pyRef = py::cast(unownedContextWrapper);
    assert(pyRef && "cast to py::object failed");
    liveContexts[context.ptr] = unownedContextWrapper;
    return PyMlirContextRef(unownedContextWrapper, std::move(pyRef));
  }
  // Use existing.
  py::object pyRef = py::cast(it->second);
  return PyMlirContextRef(it->second, std::move(pyRef));
}

PyMlirContext::LiveContextMap &PyMlirContext::getLiveContexts() {
  static LiveContextMap liveContexts;
  return liveContexts;
}

size_t PyMlirContext::getLiveCount() { return getLiveContexts().size(); }

size_t PyMlirContext::getLiveOperationCount() { return liveOperations.size(); }

py::object PyMlirContext::createOperation(
    std::string name, PyLocation location,
    llvm::Optional<std::vector<PyType *>> results,
    llvm::Optional<py::dict> attributes,
    llvm::Optional<std::vector<PyBlock *>> successors, int regions) {
  llvm::SmallVector<MlirType, 4> mlirResults;
  llvm::SmallVector<MlirBlock, 4> mlirSuccessors;
  llvm::SmallVector<std::pair<std::string, MlirAttribute>, 4> mlirAttributes;

  // General parameter validation.
  if (regions < 0)
    throw SetPyError(PyExc_ValueError, "number of regions must be >= 0");

  // Unpack/validate results.
  if (results) {
    mlirResults.reserve(results->size());
    for (PyType *result : *results) {
      // TODO: Verify result type originate from the same context.
      if (!result)
        throw SetPyError(PyExc_ValueError, "result type cannot be None");
      mlirResults.push_back(result->type);
    }
  }
  // Unpack/validate attributes.
  if (attributes) {
    mlirAttributes.reserve(attributes->size());
    for (auto &it : *attributes) {

      auto name = it.first.cast<std::string>();
      auto &attribute = it.second.cast<PyAttribute &>();
      // TODO: Verify attribute originates from the same context.
      mlirAttributes.emplace_back(std::move(name), attribute.attr);
    }
  }
  // Unpack/validate successors.
  if (successors) {
    llvm::SmallVector<MlirBlock, 4> mlirSuccessors;
    mlirSuccessors.reserve(successors->size());
    for (auto *successor : *successors) {
      // TODO: Verify successor originate from the same context.
      if (!successor)
        throw SetPyError(PyExc_ValueError, "successor block cannot be None");
      mlirSuccessors.push_back(successor->get());
    }
  }

  // Apply unpacked/validated to the operation state. Beyond this
  // point, exceptions cannot be thrown or else the state will leak.
  MlirOperationState state = mlirOperationStateGet(name.c_str(), location.loc);
  if (!mlirResults.empty())
    mlirOperationStateAddResults(&state, mlirResults.size(),
                                 mlirResults.data());
  if (!mlirAttributes.empty()) {
    // Note that the attribute names directly reference bytes in
    // mlirAttributes, so that vector must not be changed from here
    // on.
    llvm::SmallVector<MlirNamedAttribute, 4> mlirNamedAttributes;
    mlirNamedAttributes.reserve(mlirAttributes.size());
    for (auto &it : mlirAttributes)
      mlirNamedAttributes.push_back(
          mlirNamedAttributeGet(it.first.c_str(), it.second));
    mlirOperationStateAddAttributes(&state, mlirNamedAttributes.size(),
                                    mlirNamedAttributes.data());
  }
  if (!mlirSuccessors.empty())
    mlirOperationStateAddSuccessors(&state, mlirSuccessors.size(),
                                    mlirSuccessors.data());
  if (regions) {
    llvm::SmallVector<MlirRegion, 4> mlirRegions;
    mlirRegions.resize(regions);
    for (int i = 0; i < regions; ++i)
      mlirRegions[i] = mlirRegionCreate();
    mlirOperationStateAddOwnedRegions(&state, mlirRegions.size(),
                                      mlirRegions.data());
  }

  // Construct the operation.
  MlirOperation operation = mlirOperationCreate(&state);
  return PyOperation::createDetached(getRef(), operation).releaseObject();
}

//------------------------------------------------------------------------------
// PyModule
//------------------------------------------------------------------------------

PyModuleRef PyModule::create(PyMlirContextRef contextRef, MlirModule module) {
  PyModule *unownedModule = new PyModule(std::move(contextRef), module);
  // Note that the default return value policy on cast is automatic_reference,
  // which does not take ownership (delete will not be called).
  // Just be explicit.
  py::object pyRef =
      py::cast(unownedModule, py::return_value_policy::take_ownership);
  unownedModule->handle = pyRef;
  return PyModuleRef(unownedModule, std::move(pyRef));
}

//------------------------------------------------------------------------------
// PyOperation
//------------------------------------------------------------------------------

PyOperation::PyOperation(PyMlirContextRef contextRef, MlirOperation operation)
    : BaseContextObject(std::move(contextRef)), operation(operation) {}

PyOperation::~PyOperation() {
  auto &liveOperations = getContext()->liveOperations;
  assert(liveOperations.count(operation.ptr) == 1 &&
         "destroying operation not in live map");
  liveOperations.erase(operation.ptr);
  if (!isAttached()) {
    mlirOperationDestroy(operation);
  }
}

PyOperationRef PyOperation::createInstance(PyMlirContextRef contextRef,
                                           MlirOperation operation,
                                           py::object parentKeepAlive) {
  auto &liveOperations = contextRef->liveOperations;
  // Create.
  PyOperation *unownedOperation =
      new PyOperation(std::move(contextRef), operation);
  // Note that the default return value policy on cast is automatic_reference,
  // which does not take ownership (delete will not be called).
  // Just be explicit.
  py::object pyRef =
      py::cast(unownedOperation, py::return_value_policy::take_ownership);
  unownedOperation->handle = pyRef;
  if (parentKeepAlive) {
    unownedOperation->parentKeepAlive = std::move(parentKeepAlive);
  }
  liveOperations[operation.ptr] = std::make_pair(pyRef, unownedOperation);
  return PyOperationRef(unownedOperation, std::move(pyRef));
}

PyOperationRef PyOperation::forOperation(PyMlirContextRef contextRef,
                                         MlirOperation operation,
                                         py::object parentKeepAlive) {
  auto &liveOperations = contextRef->liveOperations;
  auto it = liveOperations.find(operation.ptr);
  if (it == liveOperations.end()) {
    // Create.
    return createInstance(std::move(contextRef), operation,
                          std::move(parentKeepAlive));
  }
  // Use existing.
  PyOperation *existing = it->second.second;
  assert(existing->parentKeepAlive.is(parentKeepAlive));
  py::object pyRef = py::reinterpret_borrow<py::object>(it->second.first);
  return PyOperationRef(existing, std::move(pyRef));
}

PyOperationRef PyOperation::createDetached(PyMlirContextRef contextRef,
                                           MlirOperation operation,
                                           py::object parentKeepAlive) {
  auto &liveOperations = contextRef->liveOperations;
  assert(liveOperations.count(operation.ptr) == 0 &&
         "cannot create detached operation that already exists");
  (void)liveOperations;

  PyOperationRef created = createInstance(std::move(contextRef), operation,
                                          std::move(parentKeepAlive));
  created->attached = false;
  return created;
}

void PyOperation::checkValid() {
  if (!valid) {
    throw SetPyError(PyExc_RuntimeError, "the operation has been invalidated");
  }
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
/// By default, attribute class hierarchies are one level deep (i.e. a
/// concrete attribute class extends PyAttribute); however, intermediate
/// python-visible base classes can be modeled by specifying a BaseTy.
template <typename DerivedTy, typename BaseTy = PyAttribute>
class PyConcreteAttribute : public BaseTy {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  using ClassTy = py::class_<DerivedTy, PyAttribute>;
  using IsAFunctionTy = int (*)(MlirAttribute);

  PyConcreteAttribute() = default;
  PyConcreteAttribute(PyMlirContextRef contextRef, MlirAttribute attr)
      : BaseTy(std::move(contextRef), attr) {}
  PyConcreteAttribute(PyAttribute &orig)
      : PyConcreteAttribute(orig.getContext(), castFrom(orig)) {}

  static MlirAttribute castFrom(PyAttribute &orig) {
    if (!DerivedTy::isaFunction(orig.attr)) {
      auto origRepr = py::repr(py::cast(orig)).cast<std::string>();
      throw SetPyError(PyExc_ValueError,
                       llvm::Twine("Cannot cast attribute to ") +
                           DerivedTy::pyClassName + " (from " + origRepr + ")");
    }
    return orig.attr;
  }

  static void bind(py::module &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName);
    cls.def(py::init<PyAttribute &>(), py::keep_alive<0, 1>());
    DerivedTy::bindDerived(cls);
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
              mlirStringAttrGet(context.get(), value.size(), &value[0]);
          return PyStringAttribute(context.getRef(), attr);
        },
        "Gets a uniqued string attribute");
    c.def_static(
        "get_typed",
        [](PyType &type, std::string value) {
          MlirAttribute attr =
              mlirStringAttrTypedGet(type.type, value.size(), &value[0]);
          return PyStringAttribute(type.getContext(), attr);
        },

        "Gets a uniqued string attribute associated to a type");
    c.def_property_readonly(
        "value",
        [](PyStringAttribute &self) {
          MlirStringRef stringRef = mlirStringAttrGetValue(self.attr);
          return py::str(stringRef.data, stringRef.length);
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
/// By default, type class hierarchies are one level deep (i.e. a
/// concrete type class extends PyType); however, intermediate python-visible
/// base classes can be modeled by specifying a BaseTy.
template <typename DerivedTy, typename BaseTy = PyType>
class PyConcreteType : public BaseTy {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  using ClassTy = py::class_<DerivedTy, BaseTy>;
  using IsAFunctionTy = int (*)(MlirType);

  PyConcreteType() = default;
  PyConcreteType(PyMlirContextRef contextRef, MlirType t)
      : BaseTy(std::move(contextRef), t) {}
  PyConcreteType(PyType &orig)
      : PyConcreteType(orig.getContext(), castFrom(orig)) {}

  static MlirType castFrom(PyType &orig) {
    if (!DerivedTy::isaFunction(orig.type)) {
      auto origRepr = py::repr(py::cast(orig)).cast<std::string>();
      throw SetPyError(PyExc_ValueError, llvm::Twine("Cannot cast type to ") +
                                             DerivedTy::pyClassName +
                                             " (from " + origRepr + ")");
    }
    return orig.type;
  }

  static void bind(py::module &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName);
    cls.def(py::init<PyType &>(), py::keep_alive<0, 1>());
    DerivedTy::bindDerived(cls);
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
          MlirType t = mlirIntegerTypeGet(context.get(), width);
          return PyIntegerType(context.getRef(), t);
        },
        "Create a signless integer type");
    c.def_static(
        "get_signed",
        [](PyMlirContext &context, unsigned width) {
          MlirType t = mlirIntegerTypeSignedGet(context.get(), width);
          return PyIntegerType(context.getRef(), t);
        },
        "Create a signed integer type");
    c.def_static(
        "get_unsigned",
        [](PyMlirContext &context, unsigned width) {
          MlirType t = mlirIntegerTypeUnsignedGet(context.get(), width);
          return PyIntegerType(context.getRef(), t);
        },
        "Create an unsigned integer type");
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
            MlirType t = mlirIndexTypeGet(context.get());
            return PyIndexType(context.getRef(), t);
          }),
          "Create a index type.");
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
            MlirType t = mlirBF16TypeGet(context.get());
            return PyBF16Type(context.getRef(), t);
          }),
          "Create a bf16 type.");
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
            MlirType t = mlirF16TypeGet(context.get());
            return PyF16Type(context.getRef(), t);
          }),
          "Create a f16 type.");
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
            MlirType t = mlirF32TypeGet(context.get());
            return PyF32Type(context.getRef(), t);
          }),
          "Create a f32 type.");
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
            MlirType t = mlirF64TypeGet(context.get());
            return PyF64Type(context.getRef(), t);
          }),
          "Create a f64 type.");
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
            MlirType t = mlirNoneTypeGet(context.get());
            return PyNoneType(context.getRef(), t);
          }),
          "Create a none type.");
  }
};

/// Complex Type subclass - ComplexType.
class PyComplexType : public PyConcreteType<PyComplexType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAComplex;
  static constexpr const char *pyClassName = "ComplexType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_complex",
        [](PyType &elementType) {
          // The element must be a floating point or integer scalar type.
          if (mlirTypeIsAIntegerOrFloat(elementType.type)) {
            MlirType t = mlirComplexTypeGet(elementType.type);
            return PyComplexType(elementType.getContext(), t);
          }
          throw SetPyError(
              PyExc_ValueError,
              llvm::Twine("invalid '") +
                  py::repr(py::cast(elementType)).cast<std::string>() +
                  "' and expected floating point or integer type.");
        },
        "Create a complex type");
    c.def_property_readonly(
        "element_type",
        [](PyComplexType &self) -> PyType {
          MlirType t = mlirComplexTypeGetElementType(self.type);
          return PyType(self.getContext(), t);
        },
        "Returns element type.");
  }
};

class PyShapedType : public PyConcreteType<PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAShaped;
  static constexpr const char *pyClassName = "ShapedType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_property_readonly(
        "element_type",
        [](PyShapedType &self) {
          MlirType t = mlirShapedTypeGetElementType(self.type);
          return PyType(self.getContext(), t);
        },
        "Returns the element type of the shaped type.");
    c.def_property_readonly(
        "has_rank",
        [](PyShapedType &self) -> bool {
          return mlirShapedTypeHasRank(self.type);
        },
        "Returns whether the given shaped type is ranked.");
    c.def_property_readonly(
        "rank",
        [](PyShapedType &self) {
          self.requireHasRank();
          return mlirShapedTypeGetRank(self.type);
        },
        "Returns the rank of the given ranked shaped type.");
    c.def_property_readonly(
        "has_static_shape",
        [](PyShapedType &self) -> bool {
          return mlirShapedTypeHasStaticShape(self.type);
        },
        "Returns whether the given shaped type has a static shape.");
    c.def(
        "is_dynamic_dim",
        [](PyShapedType &self, intptr_t dim) -> bool {
          self.requireHasRank();
          return mlirShapedTypeIsDynamicDim(self.type, dim);
        },
        "Returns whether the dim-th dimension of the given shaped type is "
        "dynamic.");
    c.def(
        "get_dim_size",
        [](PyShapedType &self, intptr_t dim) {
          self.requireHasRank();
          return mlirShapedTypeGetDimSize(self.type, dim);
        },
        "Returns the dim-th dimension of the given ranked shaped type.");
    c.def_static(
        "is_dynamic_size",
        [](int64_t size) -> bool { return mlirShapedTypeIsDynamicSize(size); },
        "Returns whether the given dimension size indicates a dynamic "
        "dimension.");
    c.def(
        "is_dynamic_stride_or_offset",
        [](PyShapedType &self, int64_t val) -> bool {
          self.requireHasRank();
          return mlirShapedTypeIsDynamicStrideOrOffset(val);
        },
        "Returns whether the given value is used as a placeholder for dynamic "
        "strides and offsets in shaped types.");
  }

private:
  void requireHasRank() {
    if (!mlirShapedTypeHasRank(type)) {
      throw SetPyError(
          PyExc_ValueError,
          "calling this method requires that the type has a rank.");
    }
  }
};

/// Vector Type subclass - VectorType.
class PyVectorType : public PyConcreteType<PyVectorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAVector;
  static constexpr const char *pyClassName = "VectorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_vector",
        // TODO: Make the location optional and create a default location.
        [](std::vector<int64_t> shape, PyType &elementType, PyLocation &loc) {
          MlirType t = mlirVectorTypeGetChecked(shape.size(), shape.data(),
                                                elementType.type, loc.loc);
          // TODO: Rework error reporting once diagnostic engine is exposed
          // in C API.
          if (mlirTypeIsNull(t)) {
            throw SetPyError(
                PyExc_ValueError,
                llvm::Twine("invalid '") +
                    py::repr(py::cast(elementType)).cast<std::string>() +
                    "' and expected floating point or integer type.");
          }
          return PyVectorType(elementType.getContext(), t);
        },
        "Create a vector type");
  }
};

/// Ranked Tensor Type subclass - RankedTensorType.
class PyRankedTensorType
    : public PyConcreteType<PyRankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsARankedTensor;
  static constexpr const char *pyClassName = "RankedTensorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_ranked_tensor",
        // TODO: Make the location optional and create a default location.
        [](std::vector<int64_t> shape, PyType &elementType, PyLocation &loc) {
          MlirType t = mlirRankedTensorTypeGetChecked(
              shape.size(), shape.data(), elementType.type, loc.loc);
          // TODO: Rework error reporting once diagnostic engine is exposed
          // in C API.
          if (mlirTypeIsNull(t)) {
            throw SetPyError(
                PyExc_ValueError,
                llvm::Twine("invalid '") +
                    py::repr(py::cast(elementType)).cast<std::string>() +
                    "' and expected floating point, integer, vector or "
                    "complex "
                    "type.");
          }
          return PyRankedTensorType(elementType.getContext(), t);
        },
        "Create a ranked tensor type");
  }
};

/// Unranked Tensor Type subclass - UnrankedTensorType.
class PyUnrankedTensorType
    : public PyConcreteType<PyUnrankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUnrankedTensor;
  static constexpr const char *pyClassName = "UnrankedTensorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_unranked_tensor",
        // TODO: Make the location optional and create a default location.
        [](PyType &elementType, PyLocation &loc) {
          MlirType t =
              mlirUnrankedTensorTypeGetChecked(elementType.type, loc.loc);
          // TODO: Rework error reporting once diagnostic engine is exposed
          // in C API.
          if (mlirTypeIsNull(t)) {
            throw SetPyError(
                PyExc_ValueError,
                llvm::Twine("invalid '") +
                    py::repr(py::cast(elementType)).cast<std::string>() +
                    "' and expected floating point, integer, vector or "
                    "complex "
                    "type.");
          }
          return PyUnrankedTensorType(elementType.getContext(), t);
        },
        "Create a unranked tensor type");
  }
};

/// Ranked MemRef Type subclass - MemRefType.
class PyMemRefType : public PyConcreteType<PyMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsARankedTensor;
  static constexpr const char *pyClassName = "MemRefType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    // TODO: Add mlirMemRefTypeGet and mlirMemRefTypeGetAffineMap binding
    // once the affine map binding is completed.
    c.def_static(
         "get_contiguous_memref",
         // TODO: Make the location optional and create a default location.
         [](PyType &elementType, std::vector<int64_t> shape,
            unsigned memorySpace, PyLocation &loc) {
           MlirType t = mlirMemRefTypeContiguousGetChecked(
               elementType.type, shape.size(), shape.data(), memorySpace,
               loc.loc);
           // TODO: Rework error reporting once diagnostic engine is exposed
           // in C API.
           if (mlirTypeIsNull(t)) {
             throw SetPyError(
                 PyExc_ValueError,
                 llvm::Twine("invalid '") +
                     py::repr(py::cast(elementType)).cast<std::string>() +
                     "' and expected floating point, integer, vector or "
                     "complex "
                     "type.");
           }
           return PyMemRefType(elementType.getContext(), t);
         },
         "Create a memref type")
        .def_property_readonly(
            "num_affine_maps",
            [](PyMemRefType &self) -> intptr_t {
              return mlirMemRefTypeGetNumAffineMaps(self.type);
            },
            "Returns the number of affine layout maps in the given MemRef "
            "type.")
        .def_property_readonly(
            "memory_space",
            [](PyMemRefType &self) -> unsigned {
              return mlirMemRefTypeGetMemorySpace(self.type);
            },
            "Returns the memory space of the given MemRef type.");
  }
};

/// Unranked MemRef Type subclass - UnrankedMemRefType.
class PyUnrankedMemRefType
    : public PyConcreteType<PyUnrankedMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUnrankedMemRef;
  static constexpr const char *pyClassName = "UnrankedMemRefType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
         "get_unranked_memref",
         // TODO: Make the location optional and create a default location.
         [](PyType &elementType, unsigned memorySpace, PyLocation &loc) {
           MlirType t = mlirUnrankedMemRefTypeGetChecked(elementType.type,
                                                         memorySpace, loc.loc);
           // TODO: Rework error reporting once diagnostic engine is exposed
           // in C API.
           if (mlirTypeIsNull(t)) {
             throw SetPyError(
                 PyExc_ValueError,
                 llvm::Twine("invalid '") +
                     py::repr(py::cast(elementType)).cast<std::string>() +
                     "' and expected floating point, integer, vector or "
                     "complex "
                     "type.");
           }
           return PyUnrankedMemRefType(elementType.getContext(), t);
         },
         "Create a unranked memref type")
        .def_property_readonly(
            "memory_space",
            [](PyUnrankedMemRefType &self) -> unsigned {
              return mlirUnrankedMemrefGetMemorySpace(self.type);
            },
            "Returns the memory space of the given Unranked MemRef type.");
  }
};

/// Tuple Type subclass - TupleType.
class PyTupleType : public PyConcreteType<PyTupleType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATuple;
  static constexpr const char *pyClassName = "TupleType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_tuple",
        [](PyMlirContext &context, py::list elementList) {
          intptr_t num = py::len(elementList);
          // Mapping py::list to SmallVector.
          SmallVector<MlirType, 4> elements;
          for (auto element : elementList)
            elements.push_back(element.cast<PyType>().type);
          MlirType t = mlirTupleTypeGet(context.get(), num, elements.data());
          return PyTupleType(context.getRef(), t);
        },
        "Create a tuple type");
    c.def(
        "get_type",
        [](PyTupleType &self, intptr_t pos) -> PyType {
          MlirType t = mlirTupleTypeGetType(self.type, pos);
          return PyType(self.getContext(), t);
        },
        "Returns the pos-th type in the tuple type.");
    c.def_property_readonly(
        "num_types",
        [](PyTupleType &self) -> intptr_t {
          return mlirTupleTypeGetNumTypes(self.type);
        },
        "Returns the number of types contained in a tuple.");
  }
};

/// Function type.
class PyFunctionType : public PyConcreteType<PyFunctionType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFunction;
  static constexpr const char *pyClassName = "FunctionType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyMlirContext &context, std::vector<PyType> inputs,
           std::vector<PyType> results) {
          SmallVector<MlirType, 4> inputsRaw(inputs.begin(), inputs.end());
          SmallVector<MlirType, 4> resultsRaw(results.begin(), results.end());
          MlirType t = mlirFunctionTypeGet(context.get(), inputsRaw.size(),
                                           inputsRaw.data(), resultsRaw.size(),
                                           resultsRaw.data());
          return PyFunctionType(context.getRef(), t);
        },
        py::arg("context"), py::arg("inputs"), py::arg("results"),
        "Gets a FunctionType from a list of input and result types");
    c.def_property_readonly(
        "inputs",
        [](PyFunctionType &self) {
          MlirType t = self.type;
          auto contextRef = self.getContext();
          py::list types;
          for (intptr_t i = 0, e = mlirFunctionTypeGetNumInputs(self.type);
               i < e; ++i) {
            types.append(PyType(contextRef, mlirFunctionTypeGetInput(t, i)));
          }
          return types;
        },
        "Returns the list of input types in the FunctionType.");
    c.def_property_readonly(
        "results",
        [](PyFunctionType &self) {
          MlirType t = self.type;
          auto contextRef = self.getContext();
          py::list types;
          for (intptr_t i = 0, e = mlirFunctionTypeGetNumResults(self.type);
               i < e; ++i) {
            types.append(PyType(contextRef, mlirFunctionTypeGetResult(t, i)));
          }
          return types;
        },
        "Returns the list of result types in the FunctionType.");
  }
};

} // namespace

//------------------------------------------------------------------------------
// Populates the pybind11 IR submodule.
//------------------------------------------------------------------------------

void mlir::python::populateIRSubmodule(py::module &m) {
  // Mapping of MlirContext
  py::class_<PyMlirContext>(m, "Context")
      .def(py::init<>(&PyMlirContext::createNewContextForInit))
      .def_static("_get_live_count", &PyMlirContext::getLiveCount)
      .def("_get_context_again",
           [](PyMlirContext &self) {
             PyMlirContextRef ref = PyMlirContext::forContext(self.get());
             return ref.releaseObject();
           })
      .def("_get_live_operation_count", &PyMlirContext::getLiveOperationCount)
      .def_property(
          "allow_unregistered_dialects",
          [](PyMlirContext &self) -> bool {
            return mlirContextGetAllowUnregisteredDialects(self.get());
          },
          [](PyMlirContext &self, bool value) {
            mlirContextSetAllowUnregisteredDialects(self.get(), value);
          })
      .def("create_operation", &PyMlirContext::createOperation, py::arg("name"),
           py::arg("location"), py::arg("results") = py::none(),
           py::arg("attributes") = py::none(),
           py::arg("successors") = py::none(), py::arg("regions") = 0,
           kContextCreateOperationDocstring)
      .def(
          "parse_module",
          [](PyMlirContext &self, const std::string moduleAsm) {
            MlirModule module =
                mlirModuleCreateParse(self.get(), moduleAsm.c_str());
            // TODO: Rework error reporting once diagnostic engine is exposed
            // in C API.
            if (mlirModuleIsNull(module)) {
              throw SetPyError(
                  PyExc_ValueError,
                  "Unable to parse module assembly (see diagnostics)");
            }
            return PyModule::create(self.getRef(), module).releaseObject();
          },
          kContextParseDocstring)
      .def(
          "parse_attr",
          [](PyMlirContext &self, std::string attrSpec) {
            MlirAttribute type =
                mlirAttributeParseGet(self.get(), attrSpec.c_str());
            // TODO: Rework error reporting once diagnostic engine is exposed
            // in C API.
            if (mlirAttributeIsNull(type)) {
              throw SetPyError(PyExc_ValueError,
                               llvm::Twine("Unable to parse attribute: '") +
                                   attrSpec + "'");
            }
            return PyAttribute(self.getRef(), type);
          },
          py::keep_alive<0, 1>())
      .def(
          "parse_type",
          [](PyMlirContext &self, std::string typeSpec) {
            MlirType type = mlirTypeParseGet(self.get(), typeSpec.c_str());
            // TODO: Rework error reporting once diagnostic engine is exposed
            // in C API.
            if (mlirTypeIsNull(type)) {
              throw SetPyError(PyExc_ValueError,
                               llvm::Twine("Unable to parse type: '") +
                                   typeSpec + "'");
            }
            return PyType(self.getRef(), type);
          },
          kContextParseTypeDocstring)
      .def(
          "get_unknown_location",
          [](PyMlirContext &self) {
            return PyLocation(self.getRef(),
                              mlirLocationUnknownGet(self.get()));
          },
          kContextGetUnknownLocationDocstring)
      .def(
          "get_file_location",
          [](PyMlirContext &self, std::string filename, int line, int col) {
            return PyLocation(self.getRef(),
                              mlirLocationFileLineColGet(
                                  self.get(), filename.c_str(), line, col));
          },
          kContextGetFileLocationDocstring, py::arg("filename"),
          py::arg("line"), py::arg("col"));

  py::class_<PyLocation>(m, "Location").def("__repr__", [](PyLocation &self) {
    PyPrintAccumulator printAccum;
    mlirLocationPrint(self.loc, printAccum.getCallback(),
                      printAccum.getUserData());
    return printAccum.join();
  });

  // Mapping of Module
  py::class_<PyModule>(m, "Module")
      .def_property_readonly(
          "operation",
          [](PyModule &self) {
            return PyOperation::forOperation(self.getContext(),
                                             mlirModuleGetOperation(self.get()),
                                             self.getRef().releaseObject())
                .releaseObject();
          },
          "Accesses the module as an operation")
      .def(
          "dump",
          [](PyModule &self) {
            mlirOperationDump(mlirModuleGetOperation(self.get()));
          },
          kDumpDocstring)
      .def(
          "__str__",
          [](PyModule &self) {
            MlirOperation operation = mlirModuleGetOperation(self.get());
            PyPrintAccumulator printAccum;
            mlirOperationPrint(operation, printAccum.getCallback(),
                               printAccum.getUserData());
            return printAccum.join();
          },
          kOperationStrDunderDocstring);

  // Mapping of Operation.
  py::class_<PyOperation>(m, "Operation")
      .def_property_readonly(
          "regions",
          [](PyOperation &self) { return PyRegionList(self.getRef()); })
      .def("__iter__",
           [](PyOperation &self) { return PyRegionIterator(self.getRef()); })
      .def(
          "__str__",
          [](PyOperation &self) {
            self.checkValid();
            PyPrintAccumulator printAccum;
            mlirOperationPrint(self.get(), printAccum.getCallback(),
                               printAccum.getUserData());
            return printAccum.join();
          },
          kTypeStrDunderDocstring);

  // Mapping of PyRegion.
  py::class_<PyRegion>(m, "Region")
      .def_property_readonly(
          "blocks",
          [](PyRegion &self) {
            return PyBlockList(self.getParentOperation(), self.get());
          },
          "Returns a forward-optimized sequence of blocks.")
      .def(
          "__iter__",
          [](PyRegion &self) {
            self.checkValid();
            MlirBlock firstBlock = mlirRegionGetFirstBlock(self.get());
            return PyBlockIterator(self.getParentOperation(), firstBlock);
          },
          "Iterates over blocks in the region.")
      .def("__eq__", [](PyRegion &self, py::object &other) {
        try {
          PyRegion *otherRegion = other.cast<PyRegion *>();
          return self.get().ptr == otherRegion->get().ptr;
        } catch (std::exception &e) {
          return false;
        }
      });

  // Mapping of PyBlock.
  py::class_<PyBlock>(m, "Block")
      .def_property_readonly(
          "operations",
          [](PyBlock &self) {
            return PyOperationList(self.getParentOperation(), self.get());
          },
          "Returns a forward-optimized sequence of operations.")
      .def(
          "__iter__",
          [](PyBlock &self) {
            self.checkValid();
            MlirOperation firstOperation =
                mlirBlockGetFirstOperation(self.get());
            return PyOperationIterator(self.getParentOperation(),
                                       firstOperation);
          },
          "Iterates over operations in the block.")
      .def("__eq__",
           [](PyBlock &self, py::object &other) {
             try {
               PyBlock *otherBlock = other.cast<PyBlock *>();
               return self.get().ptr == otherBlock->get().ptr;
             } catch (std::exception &e) {
               return false;
             }
           })
      .def(
          "__str__",
          [](PyBlock &self) {
            self.checkValid();
            PyPrintAccumulator printAccum;
            mlirBlockPrint(self.get(), printAccum.getCallback(),
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
            // TODO: When named attribute is removed/refactored, also remove
            // this constructor (it does an inefficient table lookup).
            auto contextRef = PyMlirContext::forContext(
                mlirAttributeGetContext(self.namedAttr.attribute));
            return PyAttribute(std::move(contextRef), self.namedAttr.attribute);
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
  PyComplexType::bind(m);
  PyShapedType::bind(m);
  PyVectorType::bind(m);
  PyRankedTensorType::bind(m);
  PyUnrankedTensorType::bind(m);
  PyMemRefType::bind(m);
  PyUnrankedMemRefType::bind(m);
  PyTupleType::bind(m);
  PyFunctionType::bind(m);

  // Container bindings.
  PyBlockIterator::bind(m);
  PyBlockList::bind(m);
  PyOperationIterator::bind(m);
  PyOperationList::bind(m);
  PyRegionIterator::bind(m);
  PyRegionList::bind(m);
}

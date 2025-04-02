//===- IRModules.cpp - IR Submodules of pybind module ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <utility>

#include "Globals.h"
#include "IRModule.h"
#include "NanobindUtils.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Debug.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace mlir::python;

using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

//------------------------------------------------------------------------------
// Docstrings (trivial, non-duplicated docstrings are included inline).
//------------------------------------------------------------------------------

static const char kContextParseTypeDocstring[] =
    R"(Parses the assembly form of a type.

Returns a Type object or raises an MLIRError if the type cannot be parsed.

See also: https://mlir.llvm.org/docs/LangRef/#type-system
)";

static const char kContextGetCallSiteLocationDocstring[] =
    R"(Gets a Location representing a caller and callsite)";

static const char kContextGetFileLocationDocstring[] =
    R"(Gets a Location representing a file, line and column)";

static const char kContextGetFileRangeDocstring[] =
    R"(Gets a Location representing a file, line and column range)";

static const char kContextGetFusedLocationDocstring[] =
    R"(Gets a Location representing a fused location with optional metadata)";

static const char kContextGetNameLocationDocString[] =
    R"(Gets a Location representing a named location with optional child location)";

static const char kModuleParseDocstring[] =
    R"(Parses a module's assembly format from a string.

Returns a new MlirModule or raises an MLIRError if the parsing fails.

See also: https://mlir.llvm.org/docs/LangRef/
)";

static const char kOperationCreateDocstring[] =
    R"(Creates a new operation.

Args:
  name: Operation name (e.g. "dialect.operation").
  results: Sequence of Type representing op result types.
  attributes: Dict of str:Attribute.
  successors: List of Block for the operation's successors.
  regions: Number of regions to create.
  location: A Location object (defaults to resolve from context manager).
  ip: An InsertionPoint (defaults to resolve from context manager or set to
    False to disable insertion, even with an insertion point set in the
    context manager).
  infer_type: Whether to infer result types.
Returns:
  A new "detached" Operation object. Detached operations can be added
  to blocks, which causes them to become "attached."
)";

static const char kOperationPrintDocstring[] =
    R"(Prints the assembly form of the operation to a file like object.

Args:
  file: The file like object to write to. Defaults to sys.stdout.
  binary: Whether to write bytes (True) or str (False). Defaults to False.
  large_elements_limit: Whether to elide elements attributes above this
    number of elements. Defaults to None (no limit).
  enable_debug_info: Whether to print debug/location information. Defaults
    to False.
  pretty_debug_info: Whether to format debug information for easier reading
    by a human (warning: the result is unparseable).
  print_generic_op_form: Whether to print the generic assembly forms of all
    ops. Defaults to False.
  use_local_Scope: Whether to print in a way that is more optimized for
    multi-threaded access but may not be consistent with how the overall
    module prints.
  assume_verified: By default, if not printing generic form, the verifier
    will be run and if it fails, generic form will be printed with a comment
    about failed verification. While a reasonable default for interactive use,
    for systematic use, it is often better for the caller to verify explicitly
    and report failures in a more robust fashion. Set this to True if doing this
    in order to avoid running a redundant verification. If the IR is actually
    invalid, behavior is undefined.
  skip_regions: Whether to skip printing regions. Defaults to False.
)";

static const char kOperationPrintStateDocstring[] =
    R"(Prints the assembly form of the operation to a file like object.

Args:
  file: The file like object to write to. Defaults to sys.stdout.
  binary: Whether to write bytes (True) or str (False). Defaults to False.
  state: AsmState capturing the operation numbering and flags.
)";

static const char kOperationGetAsmDocstring[] =
    R"(Gets the assembly form of the operation with all options available.

Args:
  binary: Whether to return a bytes (True) or str (False) object. Defaults to
    False.
  ... others ...: See the print() method for common keyword arguments for
    configuring the printout.
Returns:
  Either a bytes or str object, depending on the setting of the 'binary'
  argument.
)";

static const char kOperationPrintBytecodeDocstring[] =
    R"(Write the bytecode form of the operation to a file like object.

Args:
  file: The file like object to write to.
  desired_version: The version of bytecode to emit.
Returns:
  The bytecode writer status.
)";

static const char kOperationStrDunderDocstring[] =
    R"(Gets the assembly form of the operation with default options.

If more advanced control over the assembly formatting or I/O options is needed,
use the dedicated print or get_asm method, which supports keyword arguments to
customize behavior.
)";

static const char kDumpDocstring[] =
    R"(Dumps a debug representation of the object to stderr.)";

static const char kAppendBlockDocstring[] =
    R"(Appends a new block, with argument types as positional args.

Returns:
  The created block.
)";

static const char kValueDunderStrDocstring[] =
    R"(Returns the string form of the value.

If the value is a block argument, this is the assembly form of its type and the
position in the argument list. If the value is an operation result, this is
equivalent to printing the operation that produced it.
)";

static const char kGetNameAsOperand[] =
    R"(Returns the string form of value as an operand (i.e., the ValueID).
)";

static const char kValueReplaceAllUsesWithDocstring[] =
    R"(Replace all uses of value with the new value, updating anything in
the IR that uses 'self' to use the other value instead.
)";

static const char kValueReplaceAllUsesExceptDocstring[] =
    R"("Replace all uses of this value with the 'with' value, except for those
in 'exceptions'. 'exceptions' can be either a single operation or a list of
operations.
)";

//------------------------------------------------------------------------------
// Utilities.
//------------------------------------------------------------------------------

/// Helper for creating an @classmethod.
template <class Func, typename... Args>
nb::object classmethod(Func f, Args... args) {
  nb::object cf = nb::cpp_function(f, args...);
  return nb::borrow<nb::object>((PyClassMethod_New(cf.ptr())));
}

static nb::object
createCustomDialectWrapper(const std::string &dialectNamespace,
                           nb::object dialectDescriptor) {
  auto dialectClass = PyGlobals::get().lookupDialectClass(dialectNamespace);
  if (!dialectClass) {
    // Use the base class.
    return nb::cast(PyDialect(std::move(dialectDescriptor)));
  }

  // Create the custom implementation.
  return (*dialectClass)(std::move(dialectDescriptor));
}

static MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

static MlirStringRef toMlirStringRef(std::string_view s) {
  return mlirStringRefCreate(s.data(), s.size());
}

static MlirStringRef toMlirStringRef(const nb::bytes &s) {
  return mlirStringRefCreate(static_cast<const char *>(s.data()), s.size());
}

/// Create a block, using the current location context if no locations are
/// specified.
static MlirBlock createBlock(const nb::sequence &pyArgTypes,
                             const std::optional<nb::sequence> &pyArgLocs) {
  SmallVector<MlirType> argTypes;
  argTypes.reserve(nb::len(pyArgTypes));
  for (const auto &pyType : pyArgTypes)
    argTypes.push_back(nb::cast<PyType &>(pyType));

  SmallVector<MlirLocation> argLocs;
  if (pyArgLocs) {
    argLocs.reserve(nb::len(*pyArgLocs));
    for (const auto &pyLoc : *pyArgLocs)
      argLocs.push_back(nb::cast<PyLocation &>(pyLoc));
  } else if (!argTypes.empty()) {
    argLocs.assign(argTypes.size(), DefaultingPyLocation::resolve());
  }

  if (argTypes.size() != argLocs.size())
    throw nb::value_error(("Expected " + Twine(argTypes.size()) +
                           " locations, got: " + Twine(argLocs.size()))
                              .str()
                              .c_str());
  return mlirBlockCreate(argTypes.size(), argTypes.data(), argLocs.data());
}

/// Wrapper for the global LLVM debugging flag.
struct PyGlobalDebugFlag {
  static void set(nb::object &o, bool enable) {
    nb::ft_lock_guard lock(mutex);
    mlirEnableGlobalDebug(enable);
  }

  static bool get(const nb::object &) {
    nb::ft_lock_guard lock(mutex);
    return mlirIsGlobalDebugEnabled();
  }

  static void bind(nb::module_ &m) {
    // Debug flags.
    nb::class_<PyGlobalDebugFlag>(m, "_GlobalDebug")
        .def_prop_rw_static("flag", &PyGlobalDebugFlag::get,
                            &PyGlobalDebugFlag::set, "LLVM-wide debug flag")
        .def_static(
            "set_types",
            [](const std::string &type) {
              nb::ft_lock_guard lock(mutex);
              mlirSetGlobalDebugType(type.c_str());
            },
            "types"_a, "Sets specific debug types to be produced by LLVM")
        .def_static("set_types", [](const std::vector<std::string> &types) {
          std::vector<const char *> pointers;
          pointers.reserve(types.size());
          for (const std::string &str : types)
            pointers.push_back(str.c_str());
          nb::ft_lock_guard lock(mutex);
          mlirSetGlobalDebugTypes(pointers.data(), pointers.size());
        });
  }

private:
  static nb::ft_mutex mutex;
};

nb::ft_mutex PyGlobalDebugFlag::mutex;

struct PyAttrBuilderMap {
  static bool dunderContains(const std::string &attributeKind) {
    return PyGlobals::get().lookupAttributeBuilder(attributeKind).has_value();
  }
  static nb::callable dunderGetItemNamed(const std::string &attributeKind) {
    auto builder = PyGlobals::get().lookupAttributeBuilder(attributeKind);
    if (!builder)
      throw nb::key_error(attributeKind.c_str());
    return *builder;
  }
  static void dunderSetItemNamed(const std::string &attributeKind,
                                nb::callable func, bool replace) {
    PyGlobals::get().registerAttributeBuilder(attributeKind, std::move(func),
                                              replace);
  }

  static void bind(nb::module_ &m) {
    nb::class_<PyAttrBuilderMap>(m, "AttrBuilder")
        .def_static("contains", &PyAttrBuilderMap::dunderContains)
        .def_static("get", &PyAttrBuilderMap::dunderGetItemNamed)
        .def_static("insert", &PyAttrBuilderMap::dunderSetItemNamed,
                    "attribute_kind"_a, "attr_builder"_a, "replace"_a = false,
                    "Register an attribute builder for building MLIR "
                    "attributes from python values.");
  }
};

//------------------------------------------------------------------------------
// PyBlock
//------------------------------------------------------------------------------

nb::object PyBlock::getCapsule() {
  return nb::steal<nb::object>(mlirPythonBlockToCapsule(get()));
}

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
      throw nb::stop_iteration();
    }
    MlirRegion region = mlirOperationGetRegion(operation->get(), nextIndex++);
    return PyRegion(operation, region);
  }

  static void bind(nb::module_ &m) {
    nb::class_<PyRegionIterator>(m, "RegionIterator")
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

  PyRegionIterator dunderIter() {
    operation->checkValid();
    return PyRegionIterator(operation);
  }

  intptr_t dunderLen() {
    operation->checkValid();
    return mlirOperationGetNumRegions(operation->get());
  }

  PyRegion dunderGetItem(intptr_t index) {
    // dunderLen checks validity.
    if (index < 0 || index >= dunderLen()) {
      throw nb::index_error("attempt to access out of bounds region");
    }
    MlirRegion region = mlirOperationGetRegion(operation->get(), index);
    return PyRegion(operation, region);
  }

  static void bind(nb::module_ &m) {
    nb::class_<PyRegionList>(m, "RegionSequence")
        .def("__len__", &PyRegionList::dunderLen)
        .def("__iter__", &PyRegionList::dunderIter)
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
      throw nb::stop_iteration();
    }

    PyBlock returnBlock(operation, next);
    next = mlirBlockGetNextInRegion(next);
    return returnBlock;
  }

  static void bind(nb::module_ &m) {
    nb::class_<PyBlockIterator>(m, "BlockIterator")
        .def("__iter__", &PyBlockIterator::dunderIter)
        .def("__next__", &PyBlockIterator::dunderNext);
  }

private:
  PyOperationRef operation;
  MlirBlock next;
};

/// Blocks are exposed by the C-API as a forward-only linked list. In Python,
/// we present them as a more full-featured list-like container but optimize
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
      throw nb::index_error("attempt to access out of bounds block");
    }
    MlirBlock block = mlirRegionGetFirstBlock(region);
    while (!mlirBlockIsNull(block)) {
      if (index == 0) {
        return PyBlock(operation, block);
      }
      block = mlirBlockGetNextInRegion(block);
      index -= 1;
    }
    throw nb::index_error("attempt to access out of bounds block");
  }

  PyBlock appendBlock(const nb::args &pyArgTypes,
                      const std::optional<nb::sequence> &pyArgLocs) {
    operation->checkValid();
    MlirBlock block =
        createBlock(nb::cast<nb::sequence>(pyArgTypes), pyArgLocs);
    mlirRegionAppendOwnedBlock(region, block);
    return PyBlock(operation, block);
  }

  static void bind(nb::module_ &m) {
    nb::class_<PyBlockList>(m, "BlockList")
        .def("__getitem__", &PyBlockList::dunderGetItem)
        .def("__iter__", &PyBlockList::dunderIter)
        .def("__len__", &PyBlockList::dunderLen)
        .def("append", &PyBlockList::appendBlock, kAppendBlockDocstring,
             nb::arg("args"), nb::kw_only(),
             nb::arg("arg_locs") = std::nullopt);
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

  nb::object dunderNext() {
    parentOperation->checkValid();
    if (mlirOperationIsNull(next)) {
      throw nb::stop_iteration();
    }

    PyOperationRef returnOperation =
        PyOperation::forOperation(parentOperation->getContext(), next);
    next = mlirOperationGetNextInBlock(next);
    return returnOperation->createOpView();
  }

  static void bind(nb::module_ &m) {
    nb::class_<PyOperationIterator>(m, "OperationIterator")
        .def("__iter__", &PyOperationIterator::dunderIter)
        .def("__next__", &PyOperationIterator::dunderNext);
  }

private:
  PyOperationRef parentOperation;
  MlirOperation next;
};

/// Operations are exposed by the C-API as a forward-only linked list. In
/// Python, we present them as a more full-featured list-like container but
/// optimize it for forward iteration. Iterable operations are always owned
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

  nb::object dunderGetItem(intptr_t index) {
    parentOperation->checkValid();
    if (index < 0) {
      throw nb::index_error("attempt to access out of bounds operation");
    }
    MlirOperation childOp = mlirBlockGetFirstOperation(block);
    while (!mlirOperationIsNull(childOp)) {
      if (index == 0) {
        return PyOperation::forOperation(parentOperation->getContext(), childOp)
            ->createOpView();
      }
      childOp = mlirOperationGetNextInBlock(childOp);
      index -= 1;
    }
    throw nb::index_error("attempt to access out of bounds operation");
  }

  static void bind(nb::module_ &m) {
    nb::class_<PyOperationList>(m, "OperationList")
        .def("__getitem__", &PyOperationList::dunderGetItem)
        .def("__iter__", &PyOperationList::dunderIter)
        .def("__len__", &PyOperationList::dunderLen);
  }

private:
  PyOperationRef parentOperation;
  MlirBlock block;
};

class PyOpOperand {
public:
  PyOpOperand(MlirOpOperand opOperand) : opOperand(opOperand) {}

  nb::object getOwner() {
    MlirOperation owner = mlirOpOperandGetOwner(opOperand);
    PyMlirContextRef context =
        PyMlirContext::forContext(mlirOperationGetContext(owner));
    return PyOperation::forOperation(context, owner)->createOpView();
  }

  size_t getOperandNumber() { return mlirOpOperandGetOperandNumber(opOperand); }

  static void bind(nb::module_ &m) {
    nb::class_<PyOpOperand>(m, "OpOperand")
        .def_prop_ro("owner", &PyOpOperand::getOwner)
        .def_prop_ro("operand_number", &PyOpOperand::getOperandNumber);
  }

private:
  MlirOpOperand opOperand;
};

class PyOpOperandIterator {
public:
  PyOpOperandIterator(MlirOpOperand opOperand) : opOperand(opOperand) {}

  PyOpOperandIterator &dunderIter() { return *this; }

  PyOpOperand dunderNext() {
    if (mlirOpOperandIsNull(opOperand))
      throw nb::stop_iteration();

    PyOpOperand returnOpOperand(opOperand);
    opOperand = mlirOpOperandGetNextUse(opOperand);
    return returnOpOperand;
  }

  static void bind(nb::module_ &m) {
    nb::class_<PyOpOperandIterator>(m, "OpOperandIterator")
        .def("__iter__", &PyOpOperandIterator::dunderIter)
        .def("__next__", &PyOpOperandIterator::dunderNext);
  }

private:
  MlirOpOperand opOperand;
};

} // namespace

//------------------------------------------------------------------------------
// PyMlirContext
//------------------------------------------------------------------------------

PyMlirContext::PyMlirContext(MlirContext context) : context(context) {
  nb::gil_scoped_acquire acquire;
  nb::ft_lock_guard lock(live_contexts_mutex);
  auto &liveContexts = getLiveContexts();
  liveContexts[context.ptr] = this;
}

PyMlirContext::~PyMlirContext() {
  // Note that the only public way to construct an instance is via the
  // forContext method, which always puts the associated handle into
  // liveContexts.
  nb::gil_scoped_acquire acquire;
  {
    nb::ft_lock_guard lock(live_contexts_mutex);
    getLiveContexts().erase(context.ptr);
  }
  mlirContextDestroy(context);
}

nb::object PyMlirContext::getCapsule() {
  return nb::steal<nb::object>(mlirPythonContextToCapsule(get()));
}

nb::object PyMlirContext::createFromCapsule(nb::object capsule) {
  MlirContext rawContext = mlirPythonCapsuleToContext(capsule.ptr());
  if (mlirContextIsNull(rawContext))
    throw nb::python_error();
  return forContext(rawContext).releaseObject();
}

PyMlirContextRef PyMlirContext::forContext(MlirContext context) {
  nb::gil_scoped_acquire acquire;
  nb::ft_lock_guard lock(live_contexts_mutex);
  auto &liveContexts = getLiveContexts();
  auto it = liveContexts.find(context.ptr);
  if (it == liveContexts.end()) {
    // Create.
    PyMlirContext *unownedContextWrapper = new PyMlirContext(context);
    nb::object pyRef = nb::cast(unownedContextWrapper);
    assert(pyRef && "cast to nb::object failed");
    liveContexts[context.ptr] = unownedContextWrapper;
    return PyMlirContextRef(unownedContextWrapper, std::move(pyRef));
  }
  // Use existing.
  nb::object pyRef = nb::cast(it->second);
  return PyMlirContextRef(it->second, std::move(pyRef));
}

nb::ft_mutex PyMlirContext::live_contexts_mutex;

PyMlirContext::LiveContextMap &PyMlirContext::getLiveContexts() {
  static LiveContextMap liveContexts;
  return liveContexts;
}

size_t PyMlirContext::getLiveCount() {
  nb::ft_lock_guard lock(live_contexts_mutex);
  return getLiveContexts().size();
}

size_t PyMlirContext::getLiveOperationCount() {
  nb::ft_lock_guard lock(liveOperationsMutex);
  return liveOperations.size();
}

std::vector<PyOperation *> PyMlirContext::getLiveOperationObjects() {
  std::vector<PyOperation *> liveObjects;
  nb::ft_lock_guard lock(liveOperationsMutex);
  for (auto &entry : liveOperations)
    liveObjects.push_back(entry.second.second);
  return liveObjects;
}

size_t PyMlirContext::clearLiveOperations() {

  LiveOperationMap operations;
  {
    nb::ft_lock_guard lock(liveOperationsMutex);
    std::swap(operations, liveOperations);
  }
  for (auto &op : operations)
    op.second.second->setInvalid();
  size_t numInvalidated = operations.size();
  return numInvalidated;
}

void PyMlirContext::clearOperation(MlirOperation op) {
  PyOperation *py_op;
  {
    nb::ft_lock_guard lock(liveOperationsMutex);
    auto it = liveOperations.find(op.ptr);
    if (it == liveOperations.end()) {
      return;
    }
    py_op = it->second.second;
    liveOperations.erase(it);
  }
  py_op->setInvalid();
}

void PyMlirContext::clearOperationsInside(PyOperationBase &op) {
  typedef struct {
    PyOperation &rootOp;
    bool rootSeen;
  } callBackData;
  callBackData data{op.getOperation(), false};
  // Mark all ops below the op that the passmanager will be rooted
  // at (but not op itself - note the preorder) as invalid.
  MlirOperationWalkCallback invalidatingCallback = [](MlirOperation op,
                                                      void *userData) {
    callBackData *data = static_cast<callBackData *>(userData);
    if (LLVM_LIKELY(data->rootSeen))
      data->rootOp.getOperation().getContext()->clearOperation(op);
    else
      data->rootSeen = true;
    return MlirWalkResult::MlirWalkResultAdvance;
  };
  mlirOperationWalk(op.getOperation(), invalidatingCallback,
                    static_cast<void *>(&data), MlirWalkPreOrder);
}
void PyMlirContext::clearOperationsInside(MlirOperation op) {
  PyOperationRef opRef = PyOperation::forOperation(getRef(), op);
  clearOperationsInside(opRef->getOperation());
}

void PyMlirContext::clearOperationAndInside(PyOperationBase &op) {
  MlirOperationWalkCallback invalidatingCallback = [](MlirOperation op,
                                                      void *userData) {
    PyMlirContextRef &contextRef = *static_cast<PyMlirContextRef *>(userData);
    contextRef->clearOperation(op);
    return MlirWalkResult::MlirWalkResultAdvance;
  };
  mlirOperationWalk(op.getOperation(), invalidatingCallback,
                    &op.getOperation().getContext(), MlirWalkPreOrder);
}

size_t PyMlirContext::getLiveModuleCount() { return liveModules.size(); }

nb::object PyMlirContext::contextEnter(nb::object context) {
  return PyThreadContextEntry::pushContext(context);
}

void PyMlirContext::contextExit(const nb::object &excType,
                                const nb::object &excVal,
                                const nb::object &excTb) {
  PyThreadContextEntry::popContext(*this);
}

nb::object PyMlirContext::attachDiagnosticHandler(nb::object callback) {
  // Note that ownership is transferred to the delete callback below by way of
  // an explicit inc_ref (borrow).
  PyDiagnosticHandler *pyHandler =
      new PyDiagnosticHandler(get(), std::move(callback));
  nb::object pyHandlerObject =
      nb::cast(pyHandler, nb::rv_policy::take_ownership);
  pyHandlerObject.inc_ref();

  // In these C callbacks, the userData is a PyDiagnosticHandler* that is
  // guaranteed to be known to pybind.
  auto handlerCallback =
      +[](MlirDiagnostic diagnostic, void *userData) -> MlirLogicalResult {
    PyDiagnostic *pyDiagnostic = new PyDiagnostic(diagnostic);
    nb::object pyDiagnosticObject =
        nb::cast(pyDiagnostic, nb::rv_policy::take_ownership);

    auto *pyHandler = static_cast<PyDiagnosticHandler *>(userData);
    bool result = false;
    {
      // Since this can be called from arbitrary C++ contexts, always get the
      // gil.
      nb::gil_scoped_acquire gil;
      try {
        result = nb::cast<bool>(pyHandler->callback(pyDiagnostic));
      } catch (std::exception &e) {
        fprintf(stderr, "MLIR Python Diagnostic handler raised exception: %s\n",
                e.what());
        pyHandler->hadError = true;
      }
    }

    pyDiagnostic->invalidate();
    return result ? mlirLogicalResultSuccess() : mlirLogicalResultFailure();
  };
  auto deleteCallback = +[](void *userData) {
    auto *pyHandler = static_cast<PyDiagnosticHandler *>(userData);
    assert(pyHandler->registeredID && "handler is not registered");
    pyHandler->registeredID.reset();

    // Decrement reference, balancing the inc_ref() above.
    nb::object pyHandlerObject = nb::cast(pyHandler, nb::rv_policy::reference);
    pyHandlerObject.dec_ref();
  };

  pyHandler->registeredID = mlirContextAttachDiagnosticHandler(
      get(), handlerCallback, static_cast<void *>(pyHandler), deleteCallback);
  return pyHandlerObject;
}

MlirLogicalResult PyMlirContext::ErrorCapture::handler(MlirDiagnostic diag,
                                                       void *userData) {
  auto *self = static_cast<ErrorCapture *>(userData);
  // Check if the context requested we emit errors instead of capturing them.
  if (self->ctx->emitErrorDiagnostics)
    return mlirLogicalResultFailure();

  if (mlirDiagnosticGetSeverity(diag) != MlirDiagnosticError)
    return mlirLogicalResultFailure();

  self->errors.emplace_back(PyDiagnostic(diag).getInfo());
  return mlirLogicalResultSuccess();
}

PyMlirContext &DefaultingPyMlirContext::resolve() {
  PyMlirContext *context = PyThreadContextEntry::getDefaultContext();
  if (!context) {
    throw std::runtime_error(
        "An MLIR function requires a Context but none was provided in the call "
        "or from the surrounding environment. Either pass to the function with "
        "a 'context=' argument or establish a default using 'with Context():'");
  }
  return *context;
}

//------------------------------------------------------------------------------
// PyThreadContextEntry management
//------------------------------------------------------------------------------

std::vector<PyThreadContextEntry> &PyThreadContextEntry::getStack() {
  static thread_local std::vector<PyThreadContextEntry> stack;
  return stack;
}

PyThreadContextEntry *PyThreadContextEntry::getTopOfStack() {
  auto &stack = getStack();
  if (stack.empty())
    return nullptr;
  return &stack.back();
}

void PyThreadContextEntry::push(FrameKind frameKind, nb::object context,
                                nb::object insertionPoint,
                                nb::object location) {
  auto &stack = getStack();
  stack.emplace_back(frameKind, std::move(context), std::move(insertionPoint),
                     std::move(location));
  // If the new stack has more than one entry and the context of the new top
  // entry matches the previous, copy the insertionPoint and location from the
  // previous entry if missing from the new top entry.
  if (stack.size() > 1) {
    auto &prev = *(stack.rbegin() + 1);
    auto &current = stack.back();
    if (current.context.is(prev.context)) {
      // Default non-context objects from the previous entry.
      if (!current.insertionPoint)
        current.insertionPoint = prev.insertionPoint;
      if (!current.location)
        current.location = prev.location;
    }
  }
}

PyMlirContext *PyThreadContextEntry::getContext() {
  if (!context)
    return nullptr;
  return nb::cast<PyMlirContext *>(context);
}

PyInsertionPoint *PyThreadContextEntry::getInsertionPoint() {
  if (!insertionPoint)
    return nullptr;
  return nb::cast<PyInsertionPoint *>(insertionPoint);
}

PyLocation *PyThreadContextEntry::getLocation() {
  if (!location)
    return nullptr;
  return nb::cast<PyLocation *>(location);
}

PyMlirContext *PyThreadContextEntry::getDefaultContext() {
  auto *tos = getTopOfStack();
  return tos ? tos->getContext() : nullptr;
}

PyInsertionPoint *PyThreadContextEntry::getDefaultInsertionPoint() {
  auto *tos = getTopOfStack();
  return tos ? tos->getInsertionPoint() : nullptr;
}

PyLocation *PyThreadContextEntry::getDefaultLocation() {
  auto *tos = getTopOfStack();
  return tos ? tos->getLocation() : nullptr;
}

nb::object PyThreadContextEntry::pushContext(nb::object context) {
  push(FrameKind::Context, /*context=*/context,
       /*insertionPoint=*/nb::object(),
       /*location=*/nb::object());
  return context;
}

void PyThreadContextEntry::popContext(PyMlirContext &context) {
  auto &stack = getStack();
  if (stack.empty())
    throw std::runtime_error("Unbalanced Context enter/exit");
  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::Context && tos.getContext() != &context)
    throw std::runtime_error("Unbalanced Context enter/exit");
  stack.pop_back();
}

nb::object
PyThreadContextEntry::pushInsertionPoint(nb::object insertionPointObj) {
  PyInsertionPoint &insertionPoint =
      nb::cast<PyInsertionPoint &>(insertionPointObj);
  nb::object contextObj =
      insertionPoint.getBlock().getParentOperation()->getContext().getObject();
  push(FrameKind::InsertionPoint,
       /*context=*/contextObj,
       /*insertionPoint=*/insertionPointObj,
       /*location=*/nb::object());
  return insertionPointObj;
}

void PyThreadContextEntry::popInsertionPoint(PyInsertionPoint &insertionPoint) {
  auto &stack = getStack();
  if (stack.empty())
    throw std::runtime_error("Unbalanced InsertionPoint enter/exit");
  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::InsertionPoint &&
      tos.getInsertionPoint() != &insertionPoint)
    throw std::runtime_error("Unbalanced InsertionPoint enter/exit");
  stack.pop_back();
}

nb::object PyThreadContextEntry::pushLocation(nb::object locationObj) {
  PyLocation &location = nb::cast<PyLocation &>(locationObj);
  nb::object contextObj = location.getContext().getObject();
  push(FrameKind::Location, /*context=*/contextObj,
       /*insertionPoint=*/nb::object(),
       /*location=*/locationObj);
  return locationObj;
}

void PyThreadContextEntry::popLocation(PyLocation &location) {
  auto &stack = getStack();
  if (stack.empty())
    throw std::runtime_error("Unbalanced Location enter/exit");
  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::Location && tos.getLocation() != &location)
    throw std::runtime_error("Unbalanced Location enter/exit");
  stack.pop_back();
}

//------------------------------------------------------------------------------
// PyDiagnostic*
//------------------------------------------------------------------------------

void PyDiagnostic::invalidate() {
  valid = false;
  if (materializedNotes) {
    for (nb::handle noteObject : *materializedNotes) {
      PyDiagnostic *note = nb::cast<PyDiagnostic *>(noteObject);
      note->invalidate();
    }
  }
}

PyDiagnosticHandler::PyDiagnosticHandler(MlirContext context,
                                         nb::object callback)
    : context(context), callback(std::move(callback)) {}

PyDiagnosticHandler::~PyDiagnosticHandler() = default;

void PyDiagnosticHandler::detach() {
  if (!registeredID)
    return;
  MlirDiagnosticHandlerID localID = *registeredID;
  mlirContextDetachDiagnosticHandler(context, localID);
  assert(!registeredID && "should have unregistered");
  // Not strictly necessary but keeps stale pointers from being around to cause
  // issues.
  context = {nullptr};
}

void PyDiagnostic::checkValid() {
  if (!valid) {
    throw std::invalid_argument(
        "Diagnostic is invalid (used outside of callback)");
  }
}

MlirDiagnosticSeverity PyDiagnostic::getSeverity() {
  checkValid();
  return mlirDiagnosticGetSeverity(diagnostic);
}

PyLocation PyDiagnostic::getLocation() {
  checkValid();
  MlirLocation loc = mlirDiagnosticGetLocation(diagnostic);
  MlirContext context = mlirLocationGetContext(loc);
  return PyLocation(PyMlirContext::forContext(context), loc);
}

nb::str PyDiagnostic::getMessage() {
  checkValid();
  nb::object fileObject = nb::module_::import_("io").attr("StringIO")();
  PyFileAccumulator accum(fileObject, /*binary=*/false);
  mlirDiagnosticPrint(diagnostic, accum.getCallback(), accum.getUserData());
  return nb::cast<nb::str>(fileObject.attr("getvalue")());
}

nb::tuple PyDiagnostic::getNotes() {
  checkValid();
  if (materializedNotes)
    return *materializedNotes;
  intptr_t numNotes = mlirDiagnosticGetNumNotes(diagnostic);
  nb::tuple notes = nb::steal<nb::tuple>(PyTuple_New(numNotes));
  for (intptr_t i = 0; i < numNotes; ++i) {
    MlirDiagnostic noteDiag = mlirDiagnosticGetNote(diagnostic, i);
    nb::object diagnostic = nb::cast(PyDiagnostic(noteDiag));
    PyTuple_SET_ITEM(notes.ptr(), i, diagnostic.release().ptr());
  }
  materializedNotes = std::move(notes);

  return *materializedNotes;
}

PyDiagnostic::DiagnosticInfo PyDiagnostic::getInfo() {
  std::vector<DiagnosticInfo> notes;
  for (nb::handle n : getNotes())
    notes.emplace_back(nb::cast<PyDiagnostic>(n).getInfo());
  return {getSeverity(), getLocation(), nb::cast<std::string>(getMessage()),
          std::move(notes)};
}

//------------------------------------------------------------------------------
// PyDialect, PyDialectDescriptor, PyDialects, PyDialectRegistry
//------------------------------------------------------------------------------

MlirDialect PyDialects::getDialectForKey(const std::string &key,
                                         bool attrError) {
  MlirDialect dialect = mlirContextGetOrLoadDialect(getContext()->get(),
                                                    {key.data(), key.size()});
  if (mlirDialectIsNull(dialect)) {
    std::string msg = (Twine("Dialect '") + key + "' not found").str();
    if (attrError)
      throw nb::attribute_error(msg.c_str());
    throw nb::index_error(msg.c_str());
  }
  return dialect;
}

nb::object PyDialectRegistry::getCapsule() {
  return nb::steal<nb::object>(mlirPythonDialectRegistryToCapsule(*this));
}

PyDialectRegistry PyDialectRegistry::createFromCapsule(nb::object capsule) {
  MlirDialectRegistry rawRegistry =
      mlirPythonCapsuleToDialectRegistry(capsule.ptr());
  if (mlirDialectRegistryIsNull(rawRegistry))
    throw nb::python_error();
  return PyDialectRegistry(rawRegistry);
}

//------------------------------------------------------------------------------
// PyLocation
//------------------------------------------------------------------------------

nb::object PyLocation::getCapsule() {
  return nb::steal<nb::object>(mlirPythonLocationToCapsule(*this));
}

PyLocation PyLocation::createFromCapsule(nb::object capsule) {
  MlirLocation rawLoc = mlirPythonCapsuleToLocation(capsule.ptr());
  if (mlirLocationIsNull(rawLoc))
    throw nb::python_error();
  return PyLocation(PyMlirContext::forContext(mlirLocationGetContext(rawLoc)),
                    rawLoc);
}

nb::object PyLocation::contextEnter(nb::object locationObj) {
  return PyThreadContextEntry::pushLocation(locationObj);
}

void PyLocation::contextExit(const nb::object &excType,
                             const nb::object &excVal,
                             const nb::object &excTb) {
  PyThreadContextEntry::popLocation(*this);
}

PyLocation &DefaultingPyLocation::resolve() {
  auto *location = PyThreadContextEntry::getDefaultLocation();
  if (!location) {
    throw std::runtime_error(
        "An MLIR function requires a Location but none was provided in the "
        "call or from the surrounding environment. Either pass to the function "
        "with a 'loc=' argument or establish a default using 'with loc:'");
  }
  return *location;
}

//------------------------------------------------------------------------------
// PyModule
//------------------------------------------------------------------------------

PyModule::PyModule(PyMlirContextRef contextRef, MlirModule module)
    : BaseContextObject(std::move(contextRef)), module(module) {}

PyModule::~PyModule() {
  nb::gil_scoped_acquire acquire;
  auto &liveModules = getContext()->liveModules;
  assert(liveModules.count(module.ptr) == 1 &&
         "destroying module not in live map");
  liveModules.erase(module.ptr);
  mlirModuleDestroy(module);
}

PyModuleRef PyModule::forModule(MlirModule module) {
  MlirContext context = mlirModuleGetContext(module);
  PyMlirContextRef contextRef = PyMlirContext::forContext(context);

  nb::gil_scoped_acquire acquire;
  auto &liveModules = contextRef->liveModules;
  auto it = liveModules.find(module.ptr);
  if (it == liveModules.end()) {
    // Create.
    PyModule *unownedModule = new PyModule(std::move(contextRef), module);
    // Note that the default return value policy on cast is automatic_reference,
    // which does not take ownership (delete will not be called).
    // Just be explicit.
    nb::object pyRef = nb::cast(unownedModule, nb::rv_policy::take_ownership);
    unownedModule->handle = pyRef;
    liveModules[module.ptr] =
        std::make_pair(unownedModule->handle, unownedModule);
    return PyModuleRef(unownedModule, std::move(pyRef));
  }
  // Use existing.
  PyModule *existing = it->second.second;
  nb::object pyRef = nb::borrow<nb::object>(it->second.first);
  return PyModuleRef(existing, std::move(pyRef));
}

nb::object PyModule::createFromCapsule(nb::object capsule) {
  MlirModule rawModule = mlirPythonCapsuleToModule(capsule.ptr());
  if (mlirModuleIsNull(rawModule))
    throw nb::python_error();
  return forModule(rawModule).releaseObject();
}

nb::object PyModule::getCapsule() {
  return nb::steal<nb::object>(mlirPythonModuleToCapsule(get()));
}

//------------------------------------------------------------------------------
// PyOperation
//------------------------------------------------------------------------------

PyOperation::PyOperation(PyMlirContextRef contextRef, MlirOperation operation)
    : BaseContextObject(std::move(contextRef)), operation(operation) {}

PyOperation::~PyOperation() {
  // If the operation has already been invalidated there is nothing to do.
  if (!valid)
    return;

  // Otherwise, invalidate the operation and remove it from live map when it is
  // attached.
  if (isAttached()) {
    getContext()->clearOperation(*this);
  } else {
    // And destroy it when it is detached, i.e. owned by Python, in which case
    // all nested operations must be invalidated at removed from the live map as
    // well.
    erase();
  }
}

namespace {

// Constructs a new object of type T in-place on the Python heap, returning a
// PyObjectRef to it, loosely analogous to std::make_shared<T>().
template <typename T, class... Args>
PyObjectRef<T> makeObjectRef(Args &&...args) {
  nb::handle type = nb::type<T>();
  nb::object instance = nb::inst_alloc(type);
  T *ptr = nb::inst_ptr<T>(instance);
  new (ptr) T(std::forward<Args>(args)...);
  nb::inst_mark_ready(instance);
  return PyObjectRef<T>(ptr, std::move(instance));
}

} // namespace

PyOperationRef PyOperation::createInstance(PyMlirContextRef contextRef,
                                           MlirOperation operation,
                                           nb::object parentKeepAlive) {
  // Create.
  PyOperationRef unownedOperation =
      makeObjectRef<PyOperation>(std::move(contextRef), operation);
  unownedOperation->handle = unownedOperation.getObject();
  if (parentKeepAlive) {
    unownedOperation->parentKeepAlive = std::move(parentKeepAlive);
  }
  return unownedOperation;
}

PyOperationRef PyOperation::forOperation(PyMlirContextRef contextRef,
                                         MlirOperation operation,
                                         nb::object parentKeepAlive) {
  nb::ft_lock_guard lock(contextRef->liveOperationsMutex);
  auto &liveOperations = contextRef->liveOperations;
  auto it = liveOperations.find(operation.ptr);
  if (it == liveOperations.end()) {
    // Create.
    PyOperationRef result = createInstance(std::move(contextRef), operation,
                                           std::move(parentKeepAlive));
    liveOperations[operation.ptr] =
        std::make_pair(result.getObject(), result.get());
    return result;
  }
  // Use existing.
  PyOperation *existing = it->second.second;
  nb::object pyRef = nb::borrow<nb::object>(it->second.first);
  return PyOperationRef(existing, std::move(pyRef));
}

PyOperationRef PyOperation::createDetached(PyMlirContextRef contextRef,
                                           MlirOperation operation,
                                           nb::object parentKeepAlive) {
  nb::ft_lock_guard lock(contextRef->liveOperationsMutex);
  auto &liveOperations = contextRef->liveOperations;
  assert(liveOperations.count(operation.ptr) == 0 &&
         "cannot create detached operation that already exists");
  (void)liveOperations;
  PyOperationRef created = createInstance(std::move(contextRef), operation,
                                          std::move(parentKeepAlive));
  liveOperations[operation.ptr] =
      std::make_pair(created.getObject(), created.get());
  created->attached = false;
  return created;
}

PyOperationRef PyOperation::parse(PyMlirContextRef contextRef,
                                  const std::string &sourceStr,
                                  const std::string &sourceName) {
  PyMlirContext::ErrorCapture errors(contextRef);
  MlirOperation op =
      mlirOperationCreateParse(contextRef->get(), toMlirStringRef(sourceStr),
                               toMlirStringRef(sourceName));
  if (mlirOperationIsNull(op))
    throw MLIRError("Unable to parse operation assembly", errors.take());
  return PyOperation::createDetached(std::move(contextRef), op);
}

void PyOperation::checkValid() const {
  if (!valid) {
    throw std::runtime_error("the operation has been invalidated");
  }
}

void PyOperationBase::print(std::optional<int64_t> largeElementsLimit,
                            bool enableDebugInfo, bool prettyDebugInfo,
                            bool printGenericOpForm, bool useLocalScope,
                            bool assumeVerified, nb::object fileObject,
                            bool binary, bool skipRegions) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  if (fileObject.is_none())
    fileObject = nb::module_::import_("sys").attr("stdout");

  MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
  if (largeElementsLimit)
    mlirOpPrintingFlagsElideLargeElementsAttrs(flags, *largeElementsLimit);
  if (enableDebugInfo)
    mlirOpPrintingFlagsEnableDebugInfo(flags, /*enable=*/true,
                                       /*prettyForm=*/prettyDebugInfo);
  if (printGenericOpForm)
    mlirOpPrintingFlagsPrintGenericOpForm(flags);
  if (useLocalScope)
    mlirOpPrintingFlagsUseLocalScope(flags);
  if (assumeVerified)
    mlirOpPrintingFlagsAssumeVerified(flags);
  if (skipRegions)
    mlirOpPrintingFlagsSkipRegions(flags);

  PyFileAccumulator accum(fileObject, binary);
  mlirOperationPrintWithFlags(operation, flags, accum.getCallback(),
                              accum.getUserData());
  mlirOpPrintingFlagsDestroy(flags);
}

void PyOperationBase::print(PyAsmState &state, nb::object fileObject,
                            bool binary) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  if (fileObject.is_none())
    fileObject = nb::module_::import_("sys").attr("stdout");
  PyFileAccumulator accum(fileObject, binary);
  mlirOperationPrintWithState(operation, state.get(), accum.getCallback(),
                              accum.getUserData());
}

void PyOperationBase::writeBytecode(const nb::object &fileObject,
                                    std::optional<int64_t> bytecodeVersion) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  PyFileAccumulator accum(fileObject, /*binary=*/true);

  if (!bytecodeVersion.has_value())
    return mlirOperationWriteBytecode(operation, accum.getCallback(),
                                      accum.getUserData());

  MlirBytecodeWriterConfig config = mlirBytecodeWriterConfigCreate();
  mlirBytecodeWriterConfigDesiredEmitVersion(config, *bytecodeVersion);
  MlirLogicalResult res = mlirOperationWriteBytecodeWithConfig(
      operation, config, accum.getCallback(), accum.getUserData());
  mlirBytecodeWriterConfigDestroy(config);
  if (mlirLogicalResultIsFailure(res))
    throw nb::value_error((Twine("Unable to honor desired bytecode version ") +
                           Twine(*bytecodeVersion))
                              .str()
                              .c_str());
}

void PyOperationBase::walk(
    std::function<MlirWalkResult(MlirOperation)> callback,
    MlirWalkOrder walkOrder) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  struct UserData {
    std::function<MlirWalkResult(MlirOperation)> callback;
    bool gotException;
    std::string exceptionWhat;
    nb::object exceptionType;
  };
  UserData userData{callback, false, {}, {}};
  MlirOperationWalkCallback walkCallback = [](MlirOperation op,
                                              void *userData) {
    UserData *calleeUserData = static_cast<UserData *>(userData);
    try {
      return (calleeUserData->callback)(op);
    } catch (nb::python_error &e) {
      calleeUserData->gotException = true;
      calleeUserData->exceptionWhat = std::string(e.what());
      calleeUserData->exceptionType = nb::borrow(e.type());
      return MlirWalkResult::MlirWalkResultInterrupt;
    }
  };
  mlirOperationWalk(operation, walkCallback, &userData, walkOrder);
  if (userData.gotException) {
    std::string message("Exception raised in callback: ");
    message.append(userData.exceptionWhat);
    throw std::runtime_error(message);
  }
}

nb::object PyOperationBase::getAsm(bool binary,
                                   std::optional<int64_t> largeElementsLimit,
                                   bool enableDebugInfo, bool prettyDebugInfo,
                                   bool printGenericOpForm, bool useLocalScope,
                                   bool assumeVerified, bool skipRegions) {
  nb::object fileObject;
  if (binary) {
    fileObject = nb::module_::import_("io").attr("BytesIO")();
  } else {
    fileObject = nb::module_::import_("io").attr("StringIO")();
  }
  print(/*largeElementsLimit=*/largeElementsLimit,
        /*enableDebugInfo=*/enableDebugInfo,
        /*prettyDebugInfo=*/prettyDebugInfo,
        /*printGenericOpForm=*/printGenericOpForm,
        /*useLocalScope=*/useLocalScope,
        /*assumeVerified=*/assumeVerified,
        /*fileObject=*/fileObject,
        /*binary=*/binary,
        /*skipRegions=*/skipRegions);

  return fileObject.attr("getvalue")();
}

void PyOperationBase::moveAfter(PyOperationBase &other) {
  PyOperation &operation = getOperation();
  PyOperation &otherOp = other.getOperation();
  operation.checkValid();
  otherOp.checkValid();
  mlirOperationMoveAfter(operation, otherOp);
  operation.parentKeepAlive = otherOp.parentKeepAlive;
}

void PyOperationBase::moveBefore(PyOperationBase &other) {
  PyOperation &operation = getOperation();
  PyOperation &otherOp = other.getOperation();
  operation.checkValid();
  otherOp.checkValid();
  mlirOperationMoveBefore(operation, otherOp);
  operation.parentKeepAlive = otherOp.parentKeepAlive;
}

bool PyOperationBase::verify() {
  PyOperation &op = getOperation();
  PyMlirContext::ErrorCapture errors(op.getContext());
  if (!mlirOperationVerify(op.get()))
    throw MLIRError("Verification failed", errors.take());
  return true;
}

std::optional<PyOperationRef> PyOperation::getParentOperation() {
  checkValid();
  if (!isAttached())
    throw nb::value_error("Detached operations have no parent");
  MlirOperation operation = mlirOperationGetParentOperation(get());
  if (mlirOperationIsNull(operation))
    return {};
  return PyOperation::forOperation(getContext(), operation);
}

PyBlock PyOperation::getBlock() {
  checkValid();
  std::optional<PyOperationRef> parentOperation = getParentOperation();
  MlirBlock block = mlirOperationGetBlock(get());
  assert(!mlirBlockIsNull(block) && "Attached operation has null parent");
  assert(parentOperation && "Operation has no parent");
  return PyBlock{std::move(*parentOperation), block};
}

nb::object PyOperation::getCapsule() {
  checkValid();
  return nb::steal<nb::object>(mlirPythonOperationToCapsule(get()));
}

nb::object PyOperation::createFromCapsule(nb::object capsule) {
  MlirOperation rawOperation = mlirPythonCapsuleToOperation(capsule.ptr());
  if (mlirOperationIsNull(rawOperation))
    throw nb::python_error();
  MlirContext rawCtxt = mlirOperationGetContext(rawOperation);
  return forOperation(PyMlirContext::forContext(rawCtxt), rawOperation)
      .releaseObject();
}

static void maybeInsertOperation(PyOperationRef &op,
                                 const nb::object &maybeIp) {
  // InsertPoint active?
  if (!maybeIp.is(nb::cast(false))) {
    PyInsertionPoint *ip;
    if (maybeIp.is_none()) {
      ip = PyThreadContextEntry::getDefaultInsertionPoint();
    } else {
      ip = nb::cast<PyInsertionPoint *>(maybeIp);
    }
    if (ip)
      ip->insert(*op.get());
  }
}

nb::object PyOperation::create(std::string_view name,
                               std::optional<std::vector<PyType *>> results,
                               llvm::ArrayRef<MlirValue> operands,
                               std::optional<nb::dict> attributes,
                               std::optional<std::vector<PyBlock *>> successors,
                               int regions, DefaultingPyLocation location,
                               const nb::object &maybeIp, bool inferType) {
  llvm::SmallVector<MlirType, 4> mlirResults;
  llvm::SmallVector<MlirBlock, 4> mlirSuccessors;
  llvm::SmallVector<std::pair<std::string, MlirAttribute>, 4> mlirAttributes;

  // General parameter validation.
  if (regions < 0)
    throw nb::value_error("number of regions must be >= 0");

  // Unpack/validate results.
  if (results) {
    mlirResults.reserve(results->size());
    for (PyType *result : *results) {
      // TODO: Verify result type originate from the same context.
      if (!result)
        throw nb::value_error("result type cannot be None");
      mlirResults.push_back(*result);
    }
  }
  // Unpack/validate attributes.
  if (attributes) {
    mlirAttributes.reserve(attributes->size());
    for (std::pair<nb::handle, nb::handle> it : *attributes) {
      std::string key;
      try {
        key = nb::cast<std::string>(it.first);
      } catch (nb::cast_error &err) {
        std::string msg = "Invalid attribute key (not a string) when "
                          "attempting to create the operation \"" +
                          std::string(name) + "\" (" + err.what() + ")";
        throw nb::type_error(msg.c_str());
      }
      try {
        auto &attribute = nb::cast<PyAttribute &>(it.second);
        // TODO: Verify attribute originates from the same context.
        mlirAttributes.emplace_back(std::move(key), attribute);
      } catch (nb::cast_error &err) {
        std::string msg = "Invalid attribute value for the key \"" + key +
                          "\" when attempting to create the operation \"" +
                          std::string(name) + "\" (" + err.what() + ")";
        throw nb::type_error(msg.c_str());
      } catch (std::runtime_error &) {
        // This exception seems thrown when the value is "None".
        std::string msg =
            "Found an invalid (`None`?) attribute value for the key \"" + key +
            "\" when attempting to create the operation \"" +
            std::string(name) + "\"";
        throw std::runtime_error(msg);
      }
    }
  }
  // Unpack/validate successors.
  if (successors) {
    mlirSuccessors.reserve(successors->size());
    for (auto *successor : *successors) {
      // TODO: Verify successor originate from the same context.
      if (!successor)
        throw nb::value_error("successor block cannot be None");
      mlirSuccessors.push_back(successor->get());
    }
  }

  // Apply unpacked/validated to the operation state. Beyond this
  // point, exceptions cannot be thrown or else the state will leak.
  MlirOperationState state =
      mlirOperationStateGet(toMlirStringRef(name), location);
  if (!operands.empty())
    mlirOperationStateAddOperands(&state, operands.size(), operands.data());
  state.enableResultTypeInference = inferType;
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
      mlirNamedAttributes.push_back(mlirNamedAttributeGet(
          mlirIdentifierGet(mlirAttributeGetContext(it.second),
                            toMlirStringRef(it.first)),
          it.second));
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
  if (!operation.ptr)
    throw nb::value_error("Operation creation failed");
  PyOperationRef created =
      PyOperation::createDetached(location->getContext(), operation);
  maybeInsertOperation(created, maybeIp);

  return created.getObject();
}

nb::object PyOperation::clone(const nb::object &maybeIp) {
  MlirOperation clonedOperation = mlirOperationClone(operation);
  PyOperationRef cloned =
      PyOperation::createDetached(getContext(), clonedOperation);
  maybeInsertOperation(cloned, maybeIp);

  return cloned->createOpView();
}

nb::object PyOperation::createOpView() {
  checkValid();
  MlirIdentifier ident = mlirOperationGetName(get());
  MlirStringRef identStr = mlirIdentifierStr(ident);
  auto operationCls = PyGlobals::get().lookupOperationClass(
      StringRef(identStr.data, identStr.length));
  if (operationCls)
    return PyOpView::constructDerived(*operationCls, getRef().getObject());
  return nb::cast(PyOpView(getRef().getObject()));
}

void PyOperation::erase() {
  checkValid();
  getContext()->clearOperationAndInside(*this);
  mlirOperationDestroy(operation);
}

namespace {
/// CRTP base class for Python MLIR values that subclass Value and should be
/// castable from it. The value hierarchy is one level deep and is not supposed
/// to accommodate other levels unless core MLIR changes.
template <typename DerivedTy>
class PyConcreteValue : public PyValue {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  // and redefine bindDerived.
  using ClassTy = nb::class_<DerivedTy, PyValue>;
  using IsAFunctionTy = bool (*)(MlirValue);

  PyConcreteValue() = default;
  PyConcreteValue(PyOperationRef operationRef, MlirValue value)
      : PyValue(operationRef, value) {}
  PyConcreteValue(PyValue &orig)
      : PyConcreteValue(orig.getParentOperation(), castFrom(orig)) {}

  /// Attempts to cast the original value to the derived type and throws on
  /// type mismatches.
  static MlirValue castFrom(PyValue &orig) {
    if (!DerivedTy::isaFunction(orig.get())) {
      auto origRepr = nb::cast<std::string>(nb::repr(nb::cast(orig)));
      throw nb::value_error((Twine("Cannot cast value to ") +
                             DerivedTy::pyClassName + " (from " + origRepr +
                             ")")
                                .str()
                                .c_str());
    }
    return orig.get();
  }

  /// Binds the Python module objects to functions of this class.
  static void bind(nb::module_ &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName);
    cls.def(nb::init<PyValue &>(), nb::keep_alive<0, 1>(), nb::arg("value"));
    cls.def_static(
        "isinstance",
        [](PyValue &otherValue) -> bool {
          return DerivedTy::isaFunction(otherValue);
        },
        nb::arg("other_value"));
    cls.def(MLIR_PYTHON_MAYBE_DOWNCAST_ATTR,
            [](DerivedTy &self) { return self.maybeDownCast(); });
    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

} // namespace

/// Python wrapper for MlirOpResult.
class PyOpResult : public PyConcreteValue<PyOpResult> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirValueIsAOpResult;
  static constexpr const char *pyClassName = "OpResult";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("owner", [](PyOpResult &self) {
      assert(
          mlirOperationEqual(self.getParentOperation()->get(),
                             mlirOpResultGetOwner(self.get())) &&
          "expected the owner of the value in Python to match that in the IR");
      return self.getParentOperation().getObject();
    });
    c.def_prop_ro("result_number", [](PyOpResult &self) {
      return mlirOpResultGetResultNumber(self.get());
    });
  }
};

/// Returns the list of types of the values held by container.
template <typename Container>
static std::vector<MlirType> getValueTypes(Container &container,
                                           PyMlirContextRef &context) {
  std::vector<MlirType> result;
  result.reserve(container.size());
  for (int i = 0, e = container.size(); i < e; ++i) {
    result.push_back(mlirValueGetType(container.getElement(i).get()));
  }
  return result;
}

/// A list of operation results. Internally, these are stored as consecutive
/// elements, random access is cheap. The (returned) result list is associated
/// with the operation whose results these are, and thus extends the lifetime of
/// this operation.
class PyOpResultList : public Sliceable<PyOpResultList, PyOpResult> {
public:
  static constexpr const char *pyClassName = "OpResultList";
  using SliceableT = Sliceable<PyOpResultList, PyOpResult>;

  PyOpResultList(PyOperationRef operation, intptr_t startIndex = 0,
                 intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirOperationGetNumResults(operation->get())
                               : length,
                  step),
        operation(std::move(operation)) {}

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("types", [](PyOpResultList &self) {
      return getValueTypes(self, self.operation->getContext());
    });
    c.def_prop_ro("owner", [](PyOpResultList &self) {
      return self.operation->createOpView();
    });
  }

  PyOperationRef &getOperation() { return operation; }

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyOpResultList, PyOpResult>;

  intptr_t getRawNumElements() {
    operation->checkValid();
    return mlirOperationGetNumResults(operation->get());
  }

  PyOpResult getRawElement(intptr_t index) {
    PyValue value(operation, mlirOperationGetResult(operation->get(), index));
    return PyOpResult(value);
  }

  PyOpResultList slice(intptr_t startIndex, intptr_t length, intptr_t step) {
    return PyOpResultList(operation, startIndex, length, step);
  }

  PyOperationRef operation;
};

//------------------------------------------------------------------------------
// PyOpView
//------------------------------------------------------------------------------

static void populateResultTypes(StringRef name, nb::list resultTypeList,
                                const nb::object &resultSegmentSpecObj,
                                std::vector<int32_t> &resultSegmentLengths,
                                std::vector<PyType *> &resultTypes) {
  resultTypes.reserve(resultTypeList.size());
  if (resultSegmentSpecObj.is_none()) {
    // Non-variadic result unpacking.
    for (const auto &it : llvm::enumerate(resultTypeList)) {
      try {
        resultTypes.push_back(nb::cast<PyType *>(it.value()));
        if (!resultTypes.back())
          throw nb::cast_error();
      } catch (nb::cast_error &err) {
        throw nb::value_error((llvm::Twine("Result ") +
                               llvm::Twine(it.index()) + " of operation \"" +
                               name + "\" must be a Type (" + err.what() + ")")
                                  .str()
                                  .c_str());
      }
    }
  } else {
    // Sized result unpacking.
    auto resultSegmentSpec = nb::cast<std::vector<int>>(resultSegmentSpecObj);
    if (resultSegmentSpec.size() != resultTypeList.size()) {
      throw nb::value_error((llvm::Twine("Operation \"") + name +
                             "\" requires " +
                             llvm::Twine(resultSegmentSpec.size()) +
                             " result segments but was provided " +
                             llvm::Twine(resultTypeList.size()))
                                .str()
                                .c_str());
    }
    resultSegmentLengths.reserve(resultTypeList.size());
    for (const auto &it :
         llvm::enumerate(llvm::zip(resultTypeList, resultSegmentSpec))) {
      int segmentSpec = std::get<1>(it.value());
      if (segmentSpec == 1 || segmentSpec == 0) {
        // Unpack unary element.
        try {
          auto *resultType = nb::cast<PyType *>(std::get<0>(it.value()));
          if (resultType) {
            resultTypes.push_back(resultType);
            resultSegmentLengths.push_back(1);
          } else if (segmentSpec == 0) {
            // Allowed to be optional.
            resultSegmentLengths.push_back(0);
          } else {
            throw nb::value_error(
                (llvm::Twine("Result ") + llvm::Twine(it.index()) +
                 " of operation \"" + name +
                 "\" must be a Type (was None and result is not optional)")
                    .str()
                    .c_str());
          }
        } catch (nb::cast_error &err) {
          throw nb::value_error((llvm::Twine("Result ") +
                                 llvm::Twine(it.index()) + " of operation \"" +
                                 name + "\" must be a Type (" + err.what() +
                                 ")")
                                    .str()
                                    .c_str());
        }
      } else if (segmentSpec == -1) {
        // Unpack sequence by appending.
        try {
          if (std::get<0>(it.value()).is_none()) {
            // Treat it as an empty list.
            resultSegmentLengths.push_back(0);
          } else {
            // Unpack the list.
            auto segment = nb::cast<nb::sequence>(std::get<0>(it.value()));
            for (nb::handle segmentItem : segment) {
              resultTypes.push_back(nb::cast<PyType *>(segmentItem));
              if (!resultTypes.back()) {
                throw nb::type_error("contained a None item");
              }
            }
            resultSegmentLengths.push_back(nb::len(segment));
          }
        } catch (std::exception &err) {
          // NOTE: Sloppy to be using a catch-all here, but there are at least
          // three different unrelated exceptions that can be thrown in the
          // above "casts". Just keep the scope above small and catch them all.
          throw nb::value_error((llvm::Twine("Result ") +
                                 llvm::Twine(it.index()) + " of operation \"" +
                                 name + "\" must be a Sequence of Types (" +
                                 err.what() + ")")
                                    .str()
                                    .c_str());
        }
      } else {
        throw nb::value_error("Unexpected segment spec");
      }
    }
  }
}

static MlirValue getUniqueResult(MlirOperation operation) {
  auto numResults = mlirOperationGetNumResults(operation);
  if (numResults != 1) {
    auto name = mlirIdentifierStr(mlirOperationGetName(operation));
    throw nb::value_error((Twine("Cannot call .result on operation ") +
                           StringRef(name.data, name.length) + " which has " +
                           Twine(numResults) +
                           " results (it is only valid for operations with a "
                           "single result)")
                              .str()
                              .c_str());
  }
  return mlirOperationGetResult(operation, 0);
}

static MlirValue getOpResultOrValue(nb::handle operand) {
  if (operand.is_none()) {
    throw nb::value_error("contained a None item");
  }
  PyOperationBase *op;
  if (nb::try_cast<PyOperationBase *>(operand, op)) {
    return getUniqueResult(op->getOperation());
  }
  PyOpResultList *opResultList;
  if (nb::try_cast<PyOpResultList *>(operand, opResultList)) {
    return getUniqueResult(opResultList->getOperation()->get());
  }
  PyValue *value;
  if (nb::try_cast<PyValue *>(operand, value)) {
    return value->get();
  }
  throw nb::value_error("is not a Value");
}

nb::object PyOpView::buildGeneric(
    std::string_view name, std::tuple<int, bool> opRegionSpec,
    nb::object operandSegmentSpecObj, nb::object resultSegmentSpecObj,
    std::optional<nb::list> resultTypeList, nb::list operandList,
    std::optional<nb::dict> attributes,
    std::optional<std::vector<PyBlock *>> successors,
    std::optional<int> regions, DefaultingPyLocation location,
    const nb::object &maybeIp) {
  PyMlirContextRef context = location->getContext();

  // Class level operation construction metadata.
  // Operand and result segment specs are either none, which does no
  // variadic unpacking, or a list of ints with segment sizes, where each
  // element is either a positive number (typically 1 for a scalar) or -1 to
  // indicate that it is derived from the length of the same-indexed operand
  // or result (implying that it is a list at that position).
  std::vector<int32_t> operandSegmentLengths;
  std::vector<int32_t> resultSegmentLengths;

  // Validate/determine region count.
  int opMinRegionCount = std::get<0>(opRegionSpec);
  bool opHasNoVariadicRegions = std::get<1>(opRegionSpec);
  if (!regions) {
    regions = opMinRegionCount;
  }
  if (*regions < opMinRegionCount) {
    throw nb::value_error(
        (llvm::Twine("Operation \"") + name + "\" requires a minimum of " +
         llvm::Twine(opMinRegionCount) +
         " regions but was built with regions=" + llvm::Twine(*regions))
            .str()
            .c_str());
  }
  if (opHasNoVariadicRegions && *regions > opMinRegionCount) {
    throw nb::value_error(
        (llvm::Twine("Operation \"") + name + "\" requires a maximum of " +
         llvm::Twine(opMinRegionCount) +
         " regions but was built with regions=" + llvm::Twine(*regions))
            .str()
            .c_str());
  }

  // Unpack results.
  std::vector<PyType *> resultTypes;
  if (resultTypeList.has_value()) {
    populateResultTypes(name, *resultTypeList, resultSegmentSpecObj,
                        resultSegmentLengths, resultTypes);
  }

  // Unpack operands.
  llvm::SmallVector<MlirValue, 4> operands;
  operands.reserve(operands.size());
  if (operandSegmentSpecObj.is_none()) {
    // Non-sized operand unpacking.
    for (const auto &it : llvm::enumerate(operandList)) {
      try {
        operands.push_back(getOpResultOrValue(it.value()));
      } catch (nb::builtin_exception &err) {
        throw nb::value_error((llvm::Twine("Operand ") +
                               llvm::Twine(it.index()) + " of operation \"" +
                               name + "\" must be a Value (" + err.what() + ")")
                                  .str()
                                  .c_str());
      }
    }
  } else {
    // Sized operand unpacking.
    auto operandSegmentSpec = nb::cast<std::vector<int>>(operandSegmentSpecObj);
    if (operandSegmentSpec.size() != operandList.size()) {
      throw nb::value_error((llvm::Twine("Operation \"") + name +
                             "\" requires " +
                             llvm::Twine(operandSegmentSpec.size()) +
                             "operand segments but was provided " +
                             llvm::Twine(operandList.size()))
                                .str()
                                .c_str());
    }
    operandSegmentLengths.reserve(operandList.size());
    for (const auto &it :
         llvm::enumerate(llvm::zip(operandList, operandSegmentSpec))) {
      int segmentSpec = std::get<1>(it.value());
      if (segmentSpec == 1 || segmentSpec == 0) {
        // Unpack unary element.
        auto &operand = std::get<0>(it.value());
        if (!operand.is_none()) {
          try {

            operands.push_back(getOpResultOrValue(operand));
          } catch (nb::builtin_exception &err) {
            throw nb::value_error((llvm::Twine("Operand ") +
                                   llvm::Twine(it.index()) +
                                   " of operation \"" + name +
                                   "\" must be a Value (" + err.what() + ")")
                                      .str()
                                      .c_str());
          }

          operandSegmentLengths.push_back(1);
        } else if (segmentSpec == 0) {
          // Allowed to be optional.
          operandSegmentLengths.push_back(0);
        } else {
          throw nb::value_error(
              (llvm::Twine("Operand ") + llvm::Twine(it.index()) +
               " of operation \"" + name +
               "\" must be a Value (was None and operand is not optional)")
                  .str()
                  .c_str());
        }
      } else if (segmentSpec == -1) {
        // Unpack sequence by appending.
        try {
          if (std::get<0>(it.value()).is_none()) {
            // Treat it as an empty list.
            operandSegmentLengths.push_back(0);
          } else {
            // Unpack the list.
            auto segment = nb::cast<nb::sequence>(std::get<0>(it.value()));
            for (nb::handle segmentItem : segment) {
              operands.push_back(getOpResultOrValue(segmentItem));
            }
            operandSegmentLengths.push_back(nb::len(segment));
          }
        } catch (std::exception &err) {
          // NOTE: Sloppy to be using a catch-all here, but there are at least
          // three different unrelated exceptions that can be thrown in the
          // above "casts". Just keep the scope above small and catch them all.
          throw nb::value_error((llvm::Twine("Operand ") +
                                 llvm::Twine(it.index()) + " of operation \"" +
                                 name + "\" must be a Sequence of Values (" +
                                 err.what() + ")")
                                    .str()
                                    .c_str());
        }
      } else {
        throw nb::value_error("Unexpected segment spec");
      }
    }
  }

  // Merge operand/result segment lengths into attributes if needed.
  if (!operandSegmentLengths.empty() || !resultSegmentLengths.empty()) {
    // Dup.
    if (attributes) {
      attributes = nb::dict(*attributes);
    } else {
      attributes = nb::dict();
    }
    if (attributes->contains("resultSegmentSizes") ||
        attributes->contains("operandSegmentSizes")) {
      throw nb::value_error("Manually setting a 'resultSegmentSizes' or "
                            "'operandSegmentSizes' attribute is unsupported. "
                            "Use Operation.create for such low-level access.");
    }

    // Add resultSegmentSizes attribute.
    if (!resultSegmentLengths.empty()) {
      MlirAttribute segmentLengthAttr =
          mlirDenseI32ArrayGet(context->get(), resultSegmentLengths.size(),
                               resultSegmentLengths.data());
      (*attributes)["resultSegmentSizes"] =
          PyAttribute(context, segmentLengthAttr);
    }

    // Add operandSegmentSizes attribute.
    if (!operandSegmentLengths.empty()) {
      MlirAttribute segmentLengthAttr =
          mlirDenseI32ArrayGet(context->get(), operandSegmentLengths.size(),
                               operandSegmentLengths.data());
      (*attributes)["operandSegmentSizes"] =
          PyAttribute(context, segmentLengthAttr);
    }
  }

  // Delegate to create.
  return PyOperation::create(name,
                             /*results=*/std::move(resultTypes),
                             /*operands=*/std::move(operands),
                             /*attributes=*/std::move(attributes),
                             /*successors=*/std::move(successors),
                             /*regions=*/*regions, location, maybeIp,
                             !resultTypeList);
}

nb::object PyOpView::constructDerived(const nb::object &cls,
                                      const nb::object &operation) {
  nb::handle opViewType = nb::type<PyOpView>();
  nb::object instance = cls.attr("__new__")(cls);
  opViewType.attr("__init__")(instance, operation);
  return instance;
}

PyOpView::PyOpView(const nb::object &operationObject)
    // Casting through the PyOperationBase base-class and then back to the
    // Operation lets us accept any PyOperationBase subclass.
    : operation(nb::cast<PyOperationBase &>(operationObject).getOperation()),
      operationObject(operation.getRef().getObject()) {}

//------------------------------------------------------------------------------
// PyInsertionPoint.
//------------------------------------------------------------------------------

PyInsertionPoint::PyInsertionPoint(PyBlock &block) : block(block) {}

PyInsertionPoint::PyInsertionPoint(PyOperationBase &beforeOperationBase)
    : refOperation(beforeOperationBase.getOperation().getRef()),
      block((*refOperation)->getBlock()) {}

void PyInsertionPoint::insert(PyOperationBase &operationBase) {
  PyOperation &operation = operationBase.getOperation();
  if (operation.isAttached())
    throw nb::value_error(
        "Attempt to insert operation that is already attached");
  block.getParentOperation()->checkValid();
  MlirOperation beforeOp = {nullptr};
  if (refOperation) {
    // Insert before operation.
    (*refOperation)->checkValid();
    beforeOp = (*refOperation)->get();
  } else {
    // Insert at end (before null) is only valid if the block does not
    // already end in a known terminator (violating this will cause assertion
    // failures later).
    if (!mlirOperationIsNull(mlirBlockGetTerminator(block.get()))) {
      throw nb::index_error("Cannot insert operation at the end of a block "
                            "that already has a terminator. Did you mean to "
                            "use 'InsertionPoint.at_block_terminator(block)' "
                            "versus 'InsertionPoint(block)'?");
    }
  }
  mlirBlockInsertOwnedOperationBefore(block.get(), beforeOp, operation);
  operation.setAttached();
}

PyInsertionPoint PyInsertionPoint::atBlockBegin(PyBlock &block) {
  MlirOperation firstOp = mlirBlockGetFirstOperation(block.get());
  if (mlirOperationIsNull(firstOp)) {
    // Just insert at end.
    return PyInsertionPoint(block);
  }

  // Insert before first op.
  PyOperationRef firstOpRef = PyOperation::forOperation(
      block.getParentOperation()->getContext(), firstOp);
  return PyInsertionPoint{block, std::move(firstOpRef)};
}

PyInsertionPoint PyInsertionPoint::atBlockTerminator(PyBlock &block) {
  MlirOperation terminator = mlirBlockGetTerminator(block.get());
  if (mlirOperationIsNull(terminator))
    throw nb::value_error("Block has no terminator");
  PyOperationRef terminatorOpRef = PyOperation::forOperation(
      block.getParentOperation()->getContext(), terminator);
  return PyInsertionPoint{block, std::move(terminatorOpRef)};
}

nb::object PyInsertionPoint::contextEnter(nb::object insertPoint) {
  return PyThreadContextEntry::pushInsertionPoint(insertPoint);
}

void PyInsertionPoint::contextExit(const nb::object &excType,
                                   const nb::object &excVal,
                                   const nb::object &excTb) {
  PyThreadContextEntry::popInsertionPoint(*this);
}

//------------------------------------------------------------------------------
// PyAttribute.
//------------------------------------------------------------------------------

bool PyAttribute::operator==(const PyAttribute &other) const {
  return mlirAttributeEqual(attr, other.attr);
}

nb::object PyAttribute::getCapsule() {
  return nb::steal<nb::object>(mlirPythonAttributeToCapsule(*this));
}

PyAttribute PyAttribute::createFromCapsule(nb::object capsule) {
  MlirAttribute rawAttr = mlirPythonCapsuleToAttribute(capsule.ptr());
  if (mlirAttributeIsNull(rawAttr))
    throw nb::python_error();
  return PyAttribute(
      PyMlirContext::forContext(mlirAttributeGetContext(rawAttr)), rawAttr);
}

//------------------------------------------------------------------------------
// PyNamedAttribute.
//------------------------------------------------------------------------------

PyNamedAttribute::PyNamedAttribute(MlirAttribute attr, std::string ownedName)
    : ownedName(new std::string(std::move(ownedName))) {
  namedAttr = mlirNamedAttributeGet(
      mlirIdentifierGet(mlirAttributeGetContext(attr),
                        toMlirStringRef(*this->ownedName)),
      attr);
}

//------------------------------------------------------------------------------
// PyType.
//------------------------------------------------------------------------------

bool PyType::operator==(const PyType &other) const {
  return mlirTypeEqual(type, other.type);
}

nb::object PyType::getCapsule() {
  return nb::steal<nb::object>(mlirPythonTypeToCapsule(*this));
}

PyType PyType::createFromCapsule(nb::object capsule) {
  MlirType rawType = mlirPythonCapsuleToType(capsule.ptr());
  if (mlirTypeIsNull(rawType))
    throw nb::python_error();
  return PyType(PyMlirContext::forContext(mlirTypeGetContext(rawType)),
                rawType);
}

//------------------------------------------------------------------------------
// PyTypeID.
//------------------------------------------------------------------------------

nb::object PyTypeID::getCapsule() {
  return nb::steal<nb::object>(mlirPythonTypeIDToCapsule(*this));
}

PyTypeID PyTypeID::createFromCapsule(nb::object capsule) {
  MlirTypeID mlirTypeID = mlirPythonCapsuleToTypeID(capsule.ptr());
  if (mlirTypeIDIsNull(mlirTypeID))
    throw nb::python_error();
  return PyTypeID(mlirTypeID);
}
bool PyTypeID::operator==(const PyTypeID &other) const {
  return mlirTypeIDEqual(typeID, other.typeID);
}

//------------------------------------------------------------------------------
// PyValue and subclasses.
//------------------------------------------------------------------------------

nb::object PyValue::getCapsule() {
  return nb::steal<nb::object>(mlirPythonValueToCapsule(get()));
}

nb::object PyValue::maybeDownCast() {
  MlirType type = mlirValueGetType(get());
  MlirTypeID mlirTypeID = mlirTypeGetTypeID(type);
  assert(!mlirTypeIDIsNull(mlirTypeID) &&
         "mlirTypeID was expected to be non-null.");
  std::optional<nb::callable> valueCaster =
      PyGlobals::get().lookupValueCaster(mlirTypeID, mlirTypeGetDialect(type));
  // nb::rv_policy::move means use std::move to move the return value
  // contents into a new instance that will be owned by Python.
  nb::object thisObj = nb::cast(this, nb::rv_policy::move);
  if (!valueCaster)
    return thisObj;
  return valueCaster.value()(thisObj);
}

PyValue PyValue::createFromCapsule(nb::object capsule) {
  MlirValue value = mlirPythonCapsuleToValue(capsule.ptr());
  if (mlirValueIsNull(value))
    throw nb::python_error();
  MlirOperation owner;
  if (mlirValueIsAOpResult(value))
    owner = mlirOpResultGetOwner(value);
  if (mlirValueIsABlockArgument(value))
    owner = mlirBlockGetParentOperation(mlirBlockArgumentGetOwner(value));
  if (mlirOperationIsNull(owner))
    throw nb::python_error();
  MlirContext ctx = mlirOperationGetContext(owner);
  PyOperationRef ownerRef =
      PyOperation::forOperation(PyMlirContext::forContext(ctx), owner);
  return PyValue(ownerRef, value);
}

//------------------------------------------------------------------------------
// PySymbolTable.
//------------------------------------------------------------------------------

PySymbolTable::PySymbolTable(PyOperationBase &operation)
    : operation(operation.getOperation().getRef()) {
  symbolTable = mlirSymbolTableCreate(operation.getOperation().get());
  if (mlirSymbolTableIsNull(symbolTable)) {
    throw nb::type_error("Operation is not a Symbol Table.");
  }
}

nb::object PySymbolTable::dunderGetItem(const std::string &name) {
  operation->checkValid();
  MlirOperation symbol = mlirSymbolTableLookup(
      symbolTable, mlirStringRefCreate(name.data(), name.length()));
  if (mlirOperationIsNull(symbol))
    throw nb::key_error(
        ("Symbol '" + name + "' not in the symbol table.").c_str());

  return PyOperation::forOperation(operation->getContext(), symbol,
                                   operation.getObject())
      ->createOpView();
}

void PySymbolTable::erase(PyOperationBase &symbol) {
  operation->checkValid();
  symbol.getOperation().checkValid();
  mlirSymbolTableErase(symbolTable, symbol.getOperation().get());
  // The operation is also erased, so we must invalidate it. There may be Python
  // references to this operation so we don't want to delete it from the list of
  // live operations here.
  symbol.getOperation().valid = false;
}

void PySymbolTable::dunderDel(const std::string &name) {
  nb::object operation = dunderGetItem(name);
  erase(nb::cast<PyOperationBase &>(operation));
}

MlirAttribute PySymbolTable::insert(PyOperationBase &symbol) {
  operation->checkValid();
  symbol.getOperation().checkValid();
  MlirAttribute symbolAttr = mlirOperationGetAttributeByName(
      symbol.getOperation().get(), mlirSymbolTableGetSymbolAttributeName());
  if (mlirAttributeIsNull(symbolAttr))
    throw nb::value_error("Expected operation to have a symbol name.");
  return mlirSymbolTableInsert(symbolTable, symbol.getOperation().get());
}

MlirAttribute PySymbolTable::getSymbolName(PyOperationBase &symbol) {
  // Op must already be a symbol.
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  MlirStringRef attrName = mlirSymbolTableGetSymbolAttributeName();
  MlirAttribute existingNameAttr =
      mlirOperationGetAttributeByName(operation.get(), attrName);
  if (mlirAttributeIsNull(existingNameAttr))
    throw nb::value_error("Expected operation to have a symbol name.");
  return existingNameAttr;
}

void PySymbolTable::setSymbolName(PyOperationBase &symbol,
                                  const std::string &name) {
  // Op must already be a symbol.
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  MlirStringRef attrName = mlirSymbolTableGetSymbolAttributeName();
  MlirAttribute existingNameAttr =
      mlirOperationGetAttributeByName(operation.get(), attrName);
  if (mlirAttributeIsNull(existingNameAttr))
    throw nb::value_error("Expected operation to have a symbol name.");
  MlirAttribute newNameAttr =
      mlirStringAttrGet(operation.getContext()->get(), toMlirStringRef(name));
  mlirOperationSetAttributeByName(operation.get(), attrName, newNameAttr);
}

MlirAttribute PySymbolTable::getVisibility(PyOperationBase &symbol) {
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  MlirStringRef attrName = mlirSymbolTableGetVisibilityAttributeName();
  MlirAttribute existingVisAttr =
      mlirOperationGetAttributeByName(operation.get(), attrName);
  if (mlirAttributeIsNull(existingVisAttr))
    throw nb::value_error("Expected operation to have a symbol visibility.");
  return existingVisAttr;
}

void PySymbolTable::setVisibility(PyOperationBase &symbol,
                                  const std::string &visibility) {
  if (visibility != "public" && visibility != "private" &&
      visibility != "nested")
    throw nb::value_error(
        "Expected visibility to be 'public', 'private' or 'nested'");
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  MlirStringRef attrName = mlirSymbolTableGetVisibilityAttributeName();
  MlirAttribute existingVisAttr =
      mlirOperationGetAttributeByName(operation.get(), attrName);
  if (mlirAttributeIsNull(existingVisAttr))
    throw nb::value_error("Expected operation to have a symbol visibility.");
  MlirAttribute newVisAttr = mlirStringAttrGet(operation.getContext()->get(),
                                               toMlirStringRef(visibility));
  mlirOperationSetAttributeByName(operation.get(), attrName, newVisAttr);
}

void PySymbolTable::replaceAllSymbolUses(const std::string &oldSymbol,
                                         const std::string &newSymbol,
                                         PyOperationBase &from) {
  PyOperation &fromOperation = from.getOperation();
  fromOperation.checkValid();
  if (mlirLogicalResultIsFailure(mlirSymbolTableReplaceAllSymbolUses(
          toMlirStringRef(oldSymbol), toMlirStringRef(newSymbol),
          from.getOperation())))

    throw nb::value_error("Symbol rename failed");
}

void PySymbolTable::walkSymbolTables(PyOperationBase &from,
                                     bool allSymUsesVisible,
                                     nb::object callback) {
  PyOperation &fromOperation = from.getOperation();
  fromOperation.checkValid();
  struct UserData {
    PyMlirContextRef context;
    nb::object callback;
    bool gotException;
    std::string exceptionWhat;
    nb::object exceptionType;
  };
  UserData userData{
      fromOperation.getContext(), std::move(callback), false, {}, {}};
  mlirSymbolTableWalkSymbolTables(
      fromOperation.get(), allSymUsesVisible,
      [](MlirOperation foundOp, bool isVisible, void *calleeUserDataVoid) {
        UserData *calleeUserData = static_cast<UserData *>(calleeUserDataVoid);
        auto pyFoundOp =
            PyOperation::forOperation(calleeUserData->context, foundOp);
        if (calleeUserData->gotException)
          return;
        try {
          calleeUserData->callback(pyFoundOp.getObject(), isVisible);
        } catch (nb::python_error &e) {
          calleeUserData->gotException = true;
          calleeUserData->exceptionWhat = e.what();
          calleeUserData->exceptionType = nb::borrow(e.type());
        }
      },
      static_cast<void *>(&userData));
  if (userData.gotException) {
    std::string message("Exception raised in callback: ");
    message.append(userData.exceptionWhat);
    throw std::runtime_error(message);
  }
}

namespace {

/// Python wrapper for MlirBlockArgument.
class PyBlockArgument : public PyConcreteValue<PyBlockArgument> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirValueIsABlockArgument;
  static constexpr const char *pyClassName = "BlockArgument";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("owner", [](PyBlockArgument &self) {
      return PyBlock(self.getParentOperation(),
                     mlirBlockArgumentGetOwner(self.get()));
    });
    c.def_prop_ro("arg_number", [](PyBlockArgument &self) {
      return mlirBlockArgumentGetArgNumber(self.get());
    });
    c.def(
        "set_type",
        [](PyBlockArgument &self, PyType type) {
          return mlirBlockArgumentSetType(self.get(), type);
        },
        nb::arg("type"));
  }
};

/// A list of block arguments. Internally, these are stored as consecutive
/// elements, random access is cheap. The argument list is associated with the
/// operation that contains the block (detached blocks are not allowed in
/// Python bindings) and extends its lifetime.
class PyBlockArgumentList
    : public Sliceable<PyBlockArgumentList, PyBlockArgument> {
public:
  static constexpr const char *pyClassName = "BlockArgumentList";
  using SliceableT = Sliceable<PyBlockArgumentList, PyBlockArgument>;

  PyBlockArgumentList(PyOperationRef operation, MlirBlock block,
                      intptr_t startIndex = 0, intptr_t length = -1,
                      intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirBlockGetNumArguments(block) : length,
                  step),
        operation(std::move(operation)), block(block) {}

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("types", [](PyBlockArgumentList &self) {
      return getValueTypes(self, self.operation->getContext());
    });
  }

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyBlockArgumentList, PyBlockArgument>;

  /// Returns the number of arguments in the list.
  intptr_t getRawNumElements() {
    operation->checkValid();
    return mlirBlockGetNumArguments(block);
  }

  /// Returns `pos`-the element in the list.
  PyBlockArgument getRawElement(intptr_t pos) {
    MlirValue argument = mlirBlockGetArgument(block, pos);
    return PyBlockArgument(operation, argument);
  }

  /// Returns a sublist of this list.
  PyBlockArgumentList slice(intptr_t startIndex, intptr_t length,
                            intptr_t step) {
    return PyBlockArgumentList(operation, block, startIndex, length, step);
  }

  PyOperationRef operation;
  MlirBlock block;
};

/// A list of operation operands. Internally, these are stored as consecutive
/// elements, random access is cheap. The (returned) operand list is associated
/// with the operation whose operands these are, and thus extends the lifetime
/// of this operation.
class PyOpOperandList : public Sliceable<PyOpOperandList, PyValue> {
public:
  static constexpr const char *pyClassName = "OpOperandList";
  using SliceableT = Sliceable<PyOpOperandList, PyValue>;

  PyOpOperandList(PyOperationRef operation, intptr_t startIndex = 0,
                  intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirOperationGetNumOperands(operation->get())
                               : length,
                  step),
        operation(operation) {}

  void dunderSetItem(intptr_t index, PyValue value) {
    index = wrapIndex(index);
    mlirOperationSetOperand(operation->get(), index, value.get());
  }

  static void bindDerived(ClassTy &c) {
    c.def("__setitem__", &PyOpOperandList::dunderSetItem);
  }

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyOpOperandList, PyValue>;

  intptr_t getRawNumElements() {
    operation->checkValid();
    return mlirOperationGetNumOperands(operation->get());
  }

  PyValue getRawElement(intptr_t pos) {
    MlirValue operand = mlirOperationGetOperand(operation->get(), pos);
    MlirOperation owner;
    if (mlirValueIsAOpResult(operand))
      owner = mlirOpResultGetOwner(operand);
    else if (mlirValueIsABlockArgument(operand))
      owner = mlirBlockGetParentOperation(mlirBlockArgumentGetOwner(operand));
    else
      assert(false && "Value must be an block arg or op result.");
    PyOperationRef pyOwner =
        PyOperation::forOperation(operation->getContext(), owner);
    return PyValue(pyOwner, operand);
  }

  PyOpOperandList slice(intptr_t startIndex, intptr_t length, intptr_t step) {
    return PyOpOperandList(operation, startIndex, length, step);
  }

  PyOperationRef operation;
};

/// A list of operation successors. Internally, these are stored as consecutive
/// elements, random access is cheap. The (returned) successor list is
/// associated with the operation whose successors these are, and thus extends
/// the lifetime of this operation.
class PyOpSuccessors : public Sliceable<PyOpSuccessors, PyBlock> {
public:
  static constexpr const char *pyClassName = "OpSuccessors";

  PyOpSuccessors(PyOperationRef operation, intptr_t startIndex = 0,
                 intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirOperationGetNumSuccessors(operation->get())
                               : length,
                  step),
        operation(operation) {}

  void dunderSetItem(intptr_t index, PyBlock block) {
    index = wrapIndex(index);
    mlirOperationSetSuccessor(operation->get(), index, block.get());
  }

  static void bindDerived(ClassTy &c) {
    c.def("__setitem__", &PyOpSuccessors::dunderSetItem);
  }

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyOpSuccessors, PyBlock>;

  intptr_t getRawNumElements() {
    operation->checkValid();
    return mlirOperationGetNumSuccessors(operation->get());
  }

  PyBlock getRawElement(intptr_t pos) {
    MlirBlock block = mlirOperationGetSuccessor(operation->get(), pos);
    return PyBlock(operation, block);
  }

  PyOpSuccessors slice(intptr_t startIndex, intptr_t length, intptr_t step) {
    return PyOpSuccessors(operation, startIndex, length, step);
  }

  PyOperationRef operation;
};

/// A list of operation attributes. Can be indexed by name, producing
/// attributes, or by index, producing named attributes.
class PyOpAttributeMap {
public:
  PyOpAttributeMap(PyOperationRef operation)
      : operation(std::move(operation)) {}

  MlirAttribute dunderGetItemNamed(const std::string &name) {
    MlirAttribute attr = mlirOperationGetAttributeByName(operation->get(),
                                                         toMlirStringRef(name));
    if (mlirAttributeIsNull(attr)) {
      throw nb::key_error("attempt to access a non-existent attribute");
    }
    return attr;
  }

  PyNamedAttribute dunderGetItemIndexed(intptr_t index) {
    if (index < 0 || index >= dunderLen()) {
      throw nb::index_error("attempt to access out of bounds attribute");
    }
    MlirNamedAttribute namedAttr =
        mlirOperationGetAttribute(operation->get(), index);
    return PyNamedAttribute(
        namedAttr.attribute,
        std::string(mlirIdentifierStr(namedAttr.name).data,
                    mlirIdentifierStr(namedAttr.name).length));
  }

  void dunderSetItem(const std::string &name, const PyAttribute &attr) {
    mlirOperationSetAttributeByName(operation->get(), toMlirStringRef(name),
                                    attr);
  }

  void dunderDelItem(const std::string &name) {
    int removed = mlirOperationRemoveAttributeByName(operation->get(),
                                                     toMlirStringRef(name));
    if (!removed)
      throw nb::key_error("attempt to delete a non-existent attribute");
  }

  intptr_t dunderLen() {
    return mlirOperationGetNumAttributes(operation->get());
  }

  bool dunderContains(const std::string &name) {
    return !mlirAttributeIsNull(mlirOperationGetAttributeByName(
        operation->get(), toMlirStringRef(name)));
  }

  static void bind(nb::module_ &m) {
    nb::class_<PyOpAttributeMap>(m, "OpAttributeMap")
        .def("__contains__", &PyOpAttributeMap::dunderContains)
        .def("__len__", &PyOpAttributeMap::dunderLen)
        .def("__getitem__", &PyOpAttributeMap::dunderGetItemNamed)
        .def("__getitem__", &PyOpAttributeMap::dunderGetItemIndexed)
        .def("__setitem__", &PyOpAttributeMap::dunderSetItem)
        .def("__delitem__", &PyOpAttributeMap::dunderDelItem);
  }

private:
  PyOperationRef operation;
};

} // namespace

//------------------------------------------------------------------------------
// Populates the core exports of the 'ir' submodule.
//------------------------------------------------------------------------------

void mlir::python::populateIRCore(nb::module_ &m) {
  // disable leak warnings which tend to be false positives.
  nb::set_leak_warnings(false);
  //----------------------------------------------------------------------------
  // Enums.
  //----------------------------------------------------------------------------
  nb::enum_<MlirDiagnosticSeverity>(m, "DiagnosticSeverity")
      .value("ERROR", MlirDiagnosticError)
      .value("WARNING", MlirDiagnosticWarning)
      .value("NOTE", MlirDiagnosticNote)
      .value("REMARK", MlirDiagnosticRemark);

  nb::enum_<MlirWalkOrder>(m, "WalkOrder")
      .value("PRE_ORDER", MlirWalkPreOrder)
      .value("POST_ORDER", MlirWalkPostOrder);

  nb::enum_<MlirWalkResult>(m, "WalkResult")
      .value("ADVANCE", MlirWalkResultAdvance)
      .value("INTERRUPT", MlirWalkResultInterrupt)
      .value("SKIP", MlirWalkResultSkip);

  //----------------------------------------------------------------------------
  // Mapping of Diagnostics.
  //----------------------------------------------------------------------------
  nb::class_<PyDiagnostic>(m, "Diagnostic")
      .def_prop_ro("severity", &PyDiagnostic::getSeverity)
      .def_prop_ro("location", &PyDiagnostic::getLocation)
      .def_prop_ro("message", &PyDiagnostic::getMessage)
      .def_prop_ro("notes", &PyDiagnostic::getNotes)
      .def("__str__", [](PyDiagnostic &self) -> nb::str {
        if (!self.isValid())
          return nb::str("<Invalid Diagnostic>");
        return self.getMessage();
      });

  nb::class_<PyDiagnostic::DiagnosticInfo>(m, "DiagnosticInfo")
      .def("__init__",
           [](PyDiagnostic::DiagnosticInfo &self, PyDiagnostic diag) {
             new (&self) PyDiagnostic::DiagnosticInfo(diag.getInfo());
           })
      .def_ro("severity", &PyDiagnostic::DiagnosticInfo::severity)
      .def_ro("location", &PyDiagnostic::DiagnosticInfo::location)
      .def_ro("message", &PyDiagnostic::DiagnosticInfo::message)
      .def_ro("notes", &PyDiagnostic::DiagnosticInfo::notes)
      .def("__str__",
           [](PyDiagnostic::DiagnosticInfo &self) { return self.message; });

  nb::class_<PyDiagnosticHandler>(m, "DiagnosticHandler")
      .def("detach", &PyDiagnosticHandler::detach)
      .def_prop_ro("attached", &PyDiagnosticHandler::isAttached)
      .def_prop_ro("had_error", &PyDiagnosticHandler::getHadError)
      .def("__enter__", &PyDiagnosticHandler::contextEnter)
      .def("__exit__", &PyDiagnosticHandler::contextExit,
           nb::arg("exc_type").none(), nb::arg("exc_value").none(),
           nb::arg("traceback").none());

  //----------------------------------------------------------------------------
  // Mapping of MlirContext.
  // Note that this is exported as _BaseContext. The containing, Python level
  // __init__.py will subclass it with site-specific functionality and set a
  // "Context" attribute on this module.
  //----------------------------------------------------------------------------
  nb::class_<PyMlirContext>(m, "_BaseContext")
      .def("__init__",
           [](PyMlirContext &self) {
             MlirContext context = mlirContextCreateWithThreading(false);
             new (&self) PyMlirContext(context);
           })
      .def_static("_get_live_count", &PyMlirContext::getLiveCount)
      .def("_get_context_again",
           [](PyMlirContext &self) {
             PyMlirContextRef ref = PyMlirContext::forContext(self.get());
             return ref.releaseObject();
           })
      .def("_get_live_operation_count", &PyMlirContext::getLiveOperationCount)
      .def("_get_live_operation_objects",
           &PyMlirContext::getLiveOperationObjects)
      .def("_clear_live_operations", &PyMlirContext::clearLiveOperations)
      .def("_clear_live_operations_inside",
           nb::overload_cast<MlirOperation>(
               &PyMlirContext::clearOperationsInside))
      .def("_get_live_module_count", &PyMlirContext::getLiveModuleCount)
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyMlirContext::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyMlirContext::createFromCapsule)
      .def("__enter__", &PyMlirContext::contextEnter)
      .def("__exit__", &PyMlirContext::contextExit, nb::arg("exc_type").none(),
           nb::arg("exc_value").none(), nb::arg("traceback").none())
      .def_prop_ro_static(
          "current",
          [](nb::object & /*class*/) {
            auto *context = PyThreadContextEntry::getDefaultContext();
            if (!context)
              return nb::none();
            return nb::cast(context);
          },
          "Gets the Context bound to the current thread or raises ValueError")
      .def_prop_ro(
          "dialects",
          [](PyMlirContext &self) { return PyDialects(self.getRef()); },
          "Gets a container for accessing dialects by name")
      .def_prop_ro(
          "d", [](PyMlirContext &self) { return PyDialects(self.getRef()); },
          "Alias for 'dialect'")
      .def(
          "get_dialect_descriptor",
          [=](PyMlirContext &self, std::string &name) {
            MlirDialect dialect = mlirContextGetOrLoadDialect(
                self.get(), {name.data(), name.size()});
            if (mlirDialectIsNull(dialect)) {
              throw nb::value_error(
                  (Twine("Dialect '") + name + "' not found").str().c_str());
            }
            return PyDialectDescriptor(self.getRef(), dialect);
          },
          nb::arg("dialect_name"),
          "Gets or loads a dialect by name, returning its descriptor object")
      .def_prop_rw(
          "allow_unregistered_dialects",
          [](PyMlirContext &self) -> bool {
            return mlirContextGetAllowUnregisteredDialects(self.get());
          },
          [](PyMlirContext &self, bool value) {
            mlirContextSetAllowUnregisteredDialects(self.get(), value);
          })
      .def("attach_diagnostic_handler", &PyMlirContext::attachDiagnosticHandler,
           nb::arg("callback"),
           "Attaches a diagnostic handler that will receive callbacks")
      .def(
          "enable_multithreading",
          [](PyMlirContext &self, bool enable) {
            mlirContextEnableMultithreading(self.get(), enable);
          },
          nb::arg("enable"))
      .def(
          "is_registered_operation",
          [](PyMlirContext &self, std::string &name) {
            return mlirContextIsRegisteredOperation(
                self.get(), MlirStringRef{name.data(), name.size()});
          },
          nb::arg("operation_name"))
      .def(
          "append_dialect_registry",
          [](PyMlirContext &self, PyDialectRegistry &registry) {
            mlirContextAppendDialectRegistry(self.get(), registry);
          },
          nb::arg("registry"))
      .def_prop_rw("emit_error_diagnostics", nullptr,
                   &PyMlirContext::setEmitErrorDiagnostics,
                   "Emit error diagnostics to diagnostic handlers. By default "
                   "error diagnostics are captured and reported through "
                   "MLIRError exceptions.")
      .def("load_all_available_dialects", [](PyMlirContext &self) {
        mlirContextLoadAllAvailableDialects(self.get());
      });

  //----------------------------------------------------------------------------
  // Mapping of PyDialectDescriptor
  //----------------------------------------------------------------------------
  nb::class_<PyDialectDescriptor>(m, "DialectDescriptor")
      .def_prop_ro("namespace",
                   [](PyDialectDescriptor &self) {
                     MlirStringRef ns = mlirDialectGetNamespace(self.get());
                     return nb::str(ns.data, ns.length);
                   })
      .def("__repr__", [](PyDialectDescriptor &self) {
        MlirStringRef ns = mlirDialectGetNamespace(self.get());
        std::string repr("<DialectDescriptor ");
        repr.append(ns.data, ns.length);
        repr.append(">");
        return repr;
      });

  //----------------------------------------------------------------------------
  // Mapping of PyDialects
  //----------------------------------------------------------------------------
  nb::class_<PyDialects>(m, "Dialects")
      .def("__getitem__",
           [=](PyDialects &self, std::string keyName) {
             MlirDialect dialect =
                 self.getDialectForKey(keyName, /*attrError=*/false);
             nb::object descriptor =
                 nb::cast(PyDialectDescriptor{self.getContext(), dialect});
             return createCustomDialectWrapper(keyName, std::move(descriptor));
           })
      .def("__getattr__", [=](PyDialects &self, std::string attrName) {
        MlirDialect dialect =
            self.getDialectForKey(attrName, /*attrError=*/true);
        nb::object descriptor =
            nb::cast(PyDialectDescriptor{self.getContext(), dialect});
        return createCustomDialectWrapper(attrName, std::move(descriptor));
      });

  //----------------------------------------------------------------------------
  // Mapping of PyDialect
  //----------------------------------------------------------------------------
  nb::class_<PyDialect>(m, "Dialect")
      .def(nb::init<nb::object>(), nb::arg("descriptor"))
      .def_prop_ro("descriptor",
                   [](PyDialect &self) { return self.getDescriptor(); })
      .def("__repr__", [](nb::object self) {
        auto clazz = self.attr("__class__");
        return nb::str("<Dialect ") +
               self.attr("descriptor").attr("namespace") + nb::str(" (class ") +
               clazz.attr("__module__") + nb::str(".") +
               clazz.attr("__name__") + nb::str(")>");
      });

  //----------------------------------------------------------------------------
  // Mapping of PyDialectRegistry
  //----------------------------------------------------------------------------
  nb::class_<PyDialectRegistry>(m, "DialectRegistry")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyDialectRegistry::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyDialectRegistry::createFromCapsule)
      .def(nb::init<>());

  //----------------------------------------------------------------------------
  // Mapping of Location
  //----------------------------------------------------------------------------
  nb::class_<PyLocation>(m, "Location")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyLocation::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyLocation::createFromCapsule)
      .def("__enter__", &PyLocation::contextEnter)
      .def("__exit__", &PyLocation::contextExit, nb::arg("exc_type").none(),
           nb::arg("exc_value").none(), nb::arg("traceback").none())
      .def("__eq__",
           [](PyLocation &self, PyLocation &other) -> bool {
             return mlirLocationEqual(self, other);
           })
      .def("__eq__", [](PyLocation &self, nb::object other) { return false; })
      .def_prop_ro_static(
          "current",
          [](nb::object & /*class*/) {
            auto *loc = PyThreadContextEntry::getDefaultLocation();
            if (!loc)
              throw nb::value_error("No current Location");
            return loc;
          },
          "Gets the Location bound to the current thread or raises ValueError")
      .def_static(
          "unknown",
          [](DefaultingPyMlirContext context) {
            return PyLocation(context->getRef(),
                              mlirLocationUnknownGet(context->get()));
          },
          nb::arg("context").none() = nb::none(),
          "Gets a Location representing an unknown location")
      .def_static(
          "callsite",
          [](PyLocation callee, const std::vector<PyLocation> &frames,
             DefaultingPyMlirContext context) {
            if (frames.empty())
              throw nb::value_error("No caller frames provided");
            MlirLocation caller = frames.back().get();
            for (const PyLocation &frame :
                 llvm::reverse(llvm::ArrayRef(frames).drop_back()))
              caller = mlirLocationCallSiteGet(frame.get(), caller);
            return PyLocation(context->getRef(),
                              mlirLocationCallSiteGet(callee.get(), caller));
          },
          nb::arg("callee"), nb::arg("frames"),
          nb::arg("context").none() = nb::none(),
          kContextGetCallSiteLocationDocstring)
      .def_static(
          "file",
          [](std::string filename, int line, int col,
             DefaultingPyMlirContext context) {
            return PyLocation(
                context->getRef(),
                mlirLocationFileLineColGet(
                    context->get(), toMlirStringRef(filename), line, col));
          },
          nb::arg("filename"), nb::arg("line"), nb::arg("col"),
          nb::arg("context").none() = nb::none(),
          kContextGetFileLocationDocstring)
      .def_static(
          "file",
          [](std::string filename, int startLine, int startCol, int endLine,
             int endCol, DefaultingPyMlirContext context) {
            return PyLocation(context->getRef(),
                              mlirLocationFileLineColRangeGet(
                                  context->get(), toMlirStringRef(filename),
                                  startLine, startCol, endLine, endCol));
          },
          nb::arg("filename"), nb::arg("start_line"), nb::arg("start_col"),
          nb::arg("end_line"), nb::arg("end_col"),
          nb::arg("context").none() = nb::none(), kContextGetFileRangeDocstring)
      .def_static(
          "fused",
          [](const std::vector<PyLocation> &pyLocations,
             std::optional<PyAttribute> metadata,
             DefaultingPyMlirContext context) {
            llvm::SmallVector<MlirLocation, 4> locations;
            locations.reserve(pyLocations.size());
            for (auto &pyLocation : pyLocations)
              locations.push_back(pyLocation.get());
            MlirLocation location = mlirLocationFusedGet(
                context->get(), locations.size(), locations.data(),
                metadata ? metadata->get() : MlirAttribute{0});
            return PyLocation(context->getRef(), location);
          },
          nb::arg("locations"), nb::arg("metadata").none() = nb::none(),
          nb::arg("context").none() = nb::none(),
          kContextGetFusedLocationDocstring)
      .def_static(
          "name",
          [](std::string name, std::optional<PyLocation> childLoc,
             DefaultingPyMlirContext context) {
            return PyLocation(
                context->getRef(),
                mlirLocationNameGet(
                    context->get(), toMlirStringRef(name),
                    childLoc ? childLoc->get()
                             : mlirLocationUnknownGet(context->get())));
          },
          nb::arg("name"), nb::arg("childLoc").none() = nb::none(),
          nb::arg("context").none() = nb::none(),
          kContextGetNameLocationDocString)
      .def_static(
          "from_attr",
          [](PyAttribute &attribute, DefaultingPyMlirContext context) {
            return PyLocation(context->getRef(),
                              mlirLocationFromAttribute(attribute));
          },
          nb::arg("attribute"), nb::arg("context").none() = nb::none(),
          "Gets a Location from a LocationAttr")
      .def_prop_ro(
          "context",
          [](PyLocation &self) { return self.getContext().getObject(); },
          "Context that owns the Location")
      .def_prop_ro(
          "attr",
          [](PyLocation &self) { return mlirLocationGetAttribute(self); },
          "Get the underlying LocationAttr")
      .def(
          "emit_error",
          [](PyLocation &self, std::string message) {
            mlirEmitError(self, message.c_str());
          },
          nb::arg("message"), "Emits an error at this location")
      .def("__repr__", [](PyLocation &self) {
        PyPrintAccumulator printAccum;
        mlirLocationPrint(self, printAccum.getCallback(),
                          printAccum.getUserData());
        return printAccum.join();
      });

  //----------------------------------------------------------------------------
  // Mapping of Module
  //----------------------------------------------------------------------------
  nb::class_<PyModule>(m, "Module", nb::is_weak_referenceable())
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyModule::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyModule::createFromCapsule)
      .def_static(
          "parse",
          [](const std::string &moduleAsm, DefaultingPyMlirContext context) {
            PyMlirContext::ErrorCapture errors(context->getRef());
            MlirModule module = mlirModuleCreateParse(
                context->get(), toMlirStringRef(moduleAsm));
            if (mlirModuleIsNull(module))
              throw MLIRError("Unable to parse module assembly", errors.take());
            return PyModule::forModule(module).releaseObject();
          },
          nb::arg("asm"), nb::arg("context").none() = nb::none(),
          kModuleParseDocstring)
      .def_static(
          "parse",
          [](nb::bytes moduleAsm, DefaultingPyMlirContext context) {
            PyMlirContext::ErrorCapture errors(context->getRef());
            MlirModule module = mlirModuleCreateParse(
                context->get(), toMlirStringRef(moduleAsm));
            if (mlirModuleIsNull(module))
              throw MLIRError("Unable to parse module assembly", errors.take());
            return PyModule::forModule(module).releaseObject();
          },
          nb::arg("asm"), nb::arg("context").none() = nb::none(),
          kModuleParseDocstring)
      .def_static(
          "create",
          [](DefaultingPyLocation loc) {
            MlirModule module = mlirModuleCreateEmpty(loc);
            return PyModule::forModule(module).releaseObject();
          },
          nb::arg("loc").none() = nb::none(), "Creates an empty module")
      .def_prop_ro(
          "context",
          [](PyModule &self) { return self.getContext().getObject(); },
          "Context that created the Module")
      .def_prop_ro(
          "operation",
          [](PyModule &self) {
            return PyOperation::forOperation(self.getContext(),
                                             mlirModuleGetOperation(self.get()),
                                             self.getRef().releaseObject())
                .releaseObject();
          },
          "Accesses the module as an operation")
      .def_prop_ro(
          "body",
          [](PyModule &self) {
            PyOperationRef moduleOp = PyOperation::forOperation(
                self.getContext(), mlirModuleGetOperation(self.get()),
                self.getRef().releaseObject());
            PyBlock returnBlock(moduleOp, mlirModuleGetBody(self.get()));
            return returnBlock;
          },
          "Return the block for this module")
      .def(
          "dump",
          [](PyModule &self) {
            mlirOperationDump(mlirModuleGetOperation(self.get()));
          },
          kDumpDocstring)
      .def(
          "__str__",
          [](nb::object self) {
            // Defer to the operation's __str__.
            return self.attr("operation").attr("__str__")();
          },
          kOperationStrDunderDocstring);

  //----------------------------------------------------------------------------
  // Mapping of Operation.
  //----------------------------------------------------------------------------
  nb::class_<PyOperationBase>(m, "_OperationBase")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR,
                   [](PyOperationBase &self) {
                     return self.getOperation().getCapsule();
                   })
      .def("__eq__",
           [](PyOperationBase &self, PyOperationBase &other) {
             return &self.getOperation() == &other.getOperation();
           })
      .def("__eq__",
           [](PyOperationBase &self, nb::object other) { return false; })
      .def("__hash__",
           [](PyOperationBase &self) {
             return static_cast<size_t>(llvm::hash_value(&self.getOperation()));
           })
      .def_prop_ro("attributes",
                   [](PyOperationBase &self) {
                     return PyOpAttributeMap(self.getOperation().getRef());
                   })
      .def_prop_ro(
          "context",
          [](PyOperationBase &self) {
            PyOperation &concreteOperation = self.getOperation();
            concreteOperation.checkValid();
            return concreteOperation.getContext().getObject();
          },
          "Context that owns the Operation")
      .def_prop_ro("name",
                   [](PyOperationBase &self) {
                     auto &concreteOperation = self.getOperation();
                     concreteOperation.checkValid();
                     MlirOperation operation = concreteOperation.get();
                     MlirStringRef name =
                         mlirIdentifierStr(mlirOperationGetName(operation));
                     return nb::str(name.data, name.length);
                   })
      .def_prop_ro("operands",
                   [](PyOperationBase &self) {
                     return PyOpOperandList(self.getOperation().getRef());
                   })
      .def_prop_ro("regions",
                   [](PyOperationBase &self) {
                     return PyRegionList(self.getOperation().getRef());
                   })
      .def_prop_ro(
          "results",
          [](PyOperationBase &self) {
            return PyOpResultList(self.getOperation().getRef());
          },
          "Returns the list of Operation results.")
      .def_prop_ro(
          "result",
          [](PyOperationBase &self) {
            auto &operation = self.getOperation();
            return PyOpResult(operation.getRef(), getUniqueResult(operation))
                .maybeDownCast();
          },
          "Shortcut to get an op result if it has only one (throws an error "
          "otherwise).")
      .def_prop_ro(
          "location",
          [](PyOperationBase &self) {
            PyOperation &operation = self.getOperation();
            return PyLocation(operation.getContext(),
                              mlirOperationGetLocation(operation.get()));
          },
          "Returns the source location the operation was defined or derived "
          "from.")
      .def_prop_ro("parent",
                   [](PyOperationBase &self) -> nb::object {
                     auto parent = self.getOperation().getParentOperation();
                     if (parent)
                       return parent->getObject();
                     return nb::none();
                   })
      .def(
          "__str__",
          [](PyOperationBase &self) {
            return self.getAsm(/*binary=*/false,
                               /*largeElementsLimit=*/std::nullopt,
                               /*enableDebugInfo=*/false,
                               /*prettyDebugInfo=*/false,
                               /*printGenericOpForm=*/false,
                               /*useLocalScope=*/false,
                               /*assumeVerified=*/false,
                               /*skipRegions=*/false);
          },
          "Returns the assembly form of the operation.")
      .def("print",
           nb::overload_cast<PyAsmState &, nb::object, bool>(
               &PyOperationBase::print),
           nb::arg("state"), nb::arg("file").none() = nb::none(),
           nb::arg("binary") = false, kOperationPrintStateDocstring)
      .def("print",
           nb::overload_cast<std::optional<int64_t>, bool, bool, bool, bool,
                             bool, nb::object, bool, bool>(
               &PyOperationBase::print),
           // Careful: Lots of arguments must match up with print method.
           nb::arg("large_elements_limit").none() = nb::none(),
           nb::arg("enable_debug_info") = false,
           nb::arg("pretty_debug_info") = false,
           nb::arg("print_generic_op_form") = false,
           nb::arg("use_local_scope") = false,
           nb::arg("assume_verified") = false,
           nb::arg("file").none() = nb::none(), nb::arg("binary") = false,
           nb::arg("skip_regions") = false, kOperationPrintDocstring)
      .def("write_bytecode", &PyOperationBase::writeBytecode, nb::arg("file"),
           nb::arg("desired_version").none() = nb::none(),
           kOperationPrintBytecodeDocstring)
      .def("get_asm", &PyOperationBase::getAsm,
           // Careful: Lots of arguments must match up with get_asm method.
           nb::arg("binary") = false,
           nb::arg("large_elements_limit").none() = nb::none(),
           nb::arg("enable_debug_info") = false,
           nb::arg("pretty_debug_info") = false,
           nb::arg("print_generic_op_form") = false,
           nb::arg("use_local_scope") = false,
           nb::arg("assume_verified") = false, nb::arg("skip_regions") = false,
           kOperationGetAsmDocstring)
      .def("verify", &PyOperationBase::verify,
           "Verify the operation. Raises MLIRError if verification fails, and "
           "returns true otherwise.")
      .def("move_after", &PyOperationBase::moveAfter, nb::arg("other"),
           "Puts self immediately after the other operation in its parent "
           "block.")
      .def("move_before", &PyOperationBase::moveBefore, nb::arg("other"),
           "Puts self immediately before the other operation in its parent "
           "block.")
      .def(
          "clone",
          [](PyOperationBase &self, nb::object ip) {
            return self.getOperation().clone(ip);
          },
          nb::arg("ip").none() = nb::none())
      .def(
          "detach_from_parent",
          [](PyOperationBase &self) {
            PyOperation &operation = self.getOperation();
            operation.checkValid();
            if (!operation.isAttached())
              throw nb::value_error("Detached operation has no parent.");

            operation.detachFromParent();
            return operation.createOpView();
          },
          "Detaches the operation from its parent block.")
      .def("erase", [](PyOperationBase &self) { self.getOperation().erase(); })
      .def("walk", &PyOperationBase::walk, nb::arg("callback"),
           nb::arg("walk_order") = MlirWalkPostOrder);

  nb::class_<PyOperation, PyOperationBase>(m, "Operation")
      .def_static(
          "create",
          [](std::string_view name,
             std::optional<std::vector<PyType *>> results,
             std::optional<std::vector<PyValue *>> operands,
             std::optional<nb::dict> attributes,
             std::optional<std::vector<PyBlock *>> successors, int regions,
             DefaultingPyLocation location, const nb::object &maybeIp,
             bool inferType) {
            // Unpack/validate operands.
            llvm::SmallVector<MlirValue, 4> mlirOperands;
            if (operands) {
              mlirOperands.reserve(operands->size());
              for (PyValue *operand : *operands) {
                if (!operand)
                  throw nb::value_error("operand value cannot be None");
                mlirOperands.push_back(operand->get());
              }
            }

            return PyOperation::create(name, results, mlirOperands, attributes,
                                       successors, regions, location, maybeIp,
                                       inferType);
          },
          nb::arg("name"), nb::arg("results").none() = nb::none(),
          nb::arg("operands").none() = nb::none(),
          nb::arg("attributes").none() = nb::none(),
          nb::arg("successors").none() = nb::none(), nb::arg("regions") = 0,
          nb::arg("loc").none() = nb::none(), nb::arg("ip").none() = nb::none(),
          nb::arg("infer_type") = false, kOperationCreateDocstring)
      .def_static(
          "parse",
          [](const std::string &sourceStr, const std::string &sourceName,
             DefaultingPyMlirContext context) {
            return PyOperation::parse(context->getRef(), sourceStr, sourceName)
                ->createOpView();
          },
          nb::arg("source"), nb::kw_only(), nb::arg("source_name") = "",
          nb::arg("context").none() = nb::none(),
          "Parses an operation. Supports both text assembly format and binary "
          "bytecode format.")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyOperation::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyOperation::createFromCapsule)
      .def_prop_ro("operation", [](nb::object self) { return self; })
      .def_prop_ro("opview", &PyOperation::createOpView)
      .def_prop_ro(
          "successors",
          [](PyOperationBase &self) {
            return PyOpSuccessors(self.getOperation().getRef());
          },
          "Returns the list of Operation successors.");

  auto opViewClass =
      nb::class_<PyOpView, PyOperationBase>(m, "OpView")
          .def(nb::init<nb::object>(), nb::arg("operation"))
          .def(
              "__init__",
              [](PyOpView *self, std::string_view name,
                 std::tuple<int, bool> opRegionSpec,
                 nb::object operandSegmentSpecObj,
                 nb::object resultSegmentSpecObj,
                 std::optional<nb::list> resultTypeList, nb::list operandList,
                 std::optional<nb::dict> attributes,
                 std::optional<std::vector<PyBlock *>> successors,
                 std::optional<int> regions, DefaultingPyLocation location,
                 const nb::object &maybeIp) {
                new (self) PyOpView(PyOpView::buildGeneric(
                    name, opRegionSpec, operandSegmentSpecObj,
                    resultSegmentSpecObj, resultTypeList, operandList,
                    attributes, successors, regions, location, maybeIp));
              },
              nb::arg("name"), nb::arg("opRegionSpec"),
              nb::arg("operandSegmentSpecObj").none() = nb::none(),
              nb::arg("resultSegmentSpecObj").none() = nb::none(),
              nb::arg("results").none() = nb::none(),
              nb::arg("operands").none() = nb::none(),
              nb::arg("attributes").none() = nb::none(),
              nb::arg("successors").none() = nb::none(),
              nb::arg("regions").none() = nb::none(),
              nb::arg("loc").none() = nb::none(),
              nb::arg("ip").none() = nb::none())

          .def_prop_ro("operation", &PyOpView::getOperationObject)
          .def_prop_ro("opview", [](nb::object self) { return self; })
          .def(
              "__str__",
              [](PyOpView &self) { return nb::str(self.getOperationObject()); })
          .def_prop_ro(
              "successors",
              [](PyOperationBase &self) {
                return PyOpSuccessors(self.getOperation().getRef());
              },
              "Returns the list of Operation successors.");
  opViewClass.attr("_ODS_REGIONS") = nb::make_tuple(0, true);
  opViewClass.attr("_ODS_OPERAND_SEGMENTS") = nb::none();
  opViewClass.attr("_ODS_RESULT_SEGMENTS") = nb::none();
  // It is faster to pass the operation_name, ods_regions, and
  // ods_operand_segments/ods_result_segments as arguments to the constructor,
  // rather than to access them as attributes.
  opViewClass.attr("build_generic") = classmethod(
      [](nb::handle cls, std::optional<nb::list> resultTypeList,
         nb::list operandList, std::optional<nb::dict> attributes,
         std::optional<std::vector<PyBlock *>> successors,
         std::optional<int> regions, DefaultingPyLocation location,
         const nb::object &maybeIp) {
        std::string name = nb::cast<std::string>(cls.attr("OPERATION_NAME"));
        std::tuple<int, bool> opRegionSpec =
            nb::cast<std::tuple<int, bool>>(cls.attr("_ODS_REGIONS"));
        nb::object operandSegmentSpec = cls.attr("_ODS_OPERAND_SEGMENTS");
        nb::object resultSegmentSpec = cls.attr("_ODS_RESULT_SEGMENTS");
        return PyOpView::buildGeneric(name, opRegionSpec, operandSegmentSpec,
                                      resultSegmentSpec, resultTypeList,
                                      operandList, attributes, successors,
                                      regions, location, maybeIp);
      },
      nb::arg("cls"), nb::arg("results").none() = nb::none(),
      nb::arg("operands").none() = nb::none(),
      nb::arg("attributes").none() = nb::none(),
      nb::arg("successors").none() = nb::none(),
      nb::arg("regions").none() = nb::none(),
      nb::arg("loc").none() = nb::none(), nb::arg("ip").none() = nb::none(),
      "Builds a specific, generated OpView based on class level attributes.");
  opViewClass.attr("parse") = classmethod(
      [](const nb::object &cls, const std::string &sourceStr,
         const std::string &sourceName, DefaultingPyMlirContext context) {
        PyOperationRef parsed =
            PyOperation::parse(context->getRef(), sourceStr, sourceName);

        // Check if the expected operation was parsed, and cast to to the
        // appropriate `OpView` subclass if successful.
        // NOTE: This accesses attributes that have been automatically added to
        // `OpView` subclasses, and is not intended to be used on `OpView`
        // directly.
        std::string clsOpName =
            nb::cast<std::string>(cls.attr("OPERATION_NAME"));
        MlirStringRef identifier =
            mlirIdentifierStr(mlirOperationGetName(*parsed.get()));
        std::string_view parsedOpName(identifier.data, identifier.length);
        if (clsOpName != parsedOpName)
          throw MLIRError(Twine("Expected a '") + clsOpName + "' op, got: '" +
                          parsedOpName + "'");
        return PyOpView::constructDerived(cls, parsed.getObject());
      },
      nb::arg("cls"), nb::arg("source"), nb::kw_only(),
      nb::arg("source_name") = "", nb::arg("context").none() = nb::none(),
      "Parses a specific, generated OpView based on class level attributes");

  //----------------------------------------------------------------------------
  // Mapping of PyRegion.
  //----------------------------------------------------------------------------
  nb::class_<PyRegion>(m, "Region")
      .def_prop_ro(
          "blocks",
          [](PyRegion &self) {
            return PyBlockList(self.getParentOperation(), self.get());
          },
          "Returns a forward-optimized sequence of blocks.")
      .def_prop_ro(
          "owner",
          [](PyRegion &self) {
            return self.getParentOperation()->createOpView();
          },
          "Returns the operation owning this region.")
      .def(
          "__iter__",
          [](PyRegion &self) {
            self.checkValid();
            MlirBlock firstBlock = mlirRegionGetFirstBlock(self.get());
            return PyBlockIterator(self.getParentOperation(), firstBlock);
          },
          "Iterates over blocks in the region.")
      .def("__eq__",
           [](PyRegion &self, PyRegion &other) {
             return self.get().ptr == other.get().ptr;
           })
      .def("__eq__", [](PyRegion &self, nb::object &other) { return false; });

  //----------------------------------------------------------------------------
  // Mapping of PyBlock.
  //----------------------------------------------------------------------------
  nb::class_<PyBlock>(m, "Block")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyBlock::getCapsule)
      .def_prop_ro(
          "owner",
          [](PyBlock &self) {
            return self.getParentOperation()->createOpView();
          },
          "Returns the owning operation of this block.")
      .def_prop_ro(
          "region",
          [](PyBlock &self) {
            MlirRegion region = mlirBlockGetParentRegion(self.get());
            return PyRegion(self.getParentOperation(), region);
          },
          "Returns the owning region of this block.")
      .def_prop_ro(
          "arguments",
          [](PyBlock &self) {
            return PyBlockArgumentList(self.getParentOperation(), self.get());
          },
          "Returns a list of block arguments.")
      .def(
          "add_argument",
          [](PyBlock &self, const PyType &type, const PyLocation &loc) {
            return mlirBlockAddArgument(self.get(), type, loc);
          },
          "Append an argument of the specified type to the block and returns "
          "the newly added argument.")
      .def(
          "erase_argument",
          [](PyBlock &self, unsigned index) {
            return mlirBlockEraseArgument(self.get(), index);
          },
          "Erase the argument at 'index' and remove it from the argument list.")
      .def_prop_ro(
          "operations",
          [](PyBlock &self) {
            return PyOperationList(self.getParentOperation(), self.get());
          },
          "Returns a forward-optimized sequence of operations.")
      .def_static(
          "create_at_start",
          [](PyRegion &parent, const nb::sequence &pyArgTypes,
             const std::optional<nb::sequence> &pyArgLocs) {
            parent.checkValid();
            MlirBlock block = createBlock(pyArgTypes, pyArgLocs);
            mlirRegionInsertOwnedBlock(parent, 0, block);
            return PyBlock(parent.getParentOperation(), block);
          },
          nb::arg("parent"), nb::arg("arg_types") = nb::list(),
          nb::arg("arg_locs") = std::nullopt,
          "Creates and returns a new Block at the beginning of the given "
          "region (with given argument types and locations).")
      .def(
          "append_to",
          [](PyBlock &self, PyRegion &region) {
            MlirBlock b = self.get();
            if (!mlirRegionIsNull(mlirBlockGetParentRegion(b)))
              mlirBlockDetach(b);
            mlirRegionAppendOwnedBlock(region.get(), b);
          },
          "Append this block to a region, transferring ownership if necessary")
      .def(
          "create_before",
          [](PyBlock &self, const nb::args &pyArgTypes,
             const std::optional<nb::sequence> &pyArgLocs) {
            self.checkValid();
            MlirBlock block =
                createBlock(nb::cast<nb::sequence>(pyArgTypes), pyArgLocs);
            MlirRegion region = mlirBlockGetParentRegion(self.get());
            mlirRegionInsertOwnedBlockBefore(region, self.get(), block);
            return PyBlock(self.getParentOperation(), block);
          },
          nb::arg("arg_types"), nb::kw_only(),
          nb::arg("arg_locs") = std::nullopt,
          "Creates and returns a new Block before this block "
          "(with given argument types and locations).")
      .def(
          "create_after",
          [](PyBlock &self, const nb::args &pyArgTypes,
             const std::optional<nb::sequence> &pyArgLocs) {
            self.checkValid();
            MlirBlock block =
                createBlock(nb::cast<nb::sequence>(pyArgTypes), pyArgLocs);
            MlirRegion region = mlirBlockGetParentRegion(self.get());
            mlirRegionInsertOwnedBlockAfter(region, self.get(), block);
            return PyBlock(self.getParentOperation(), block);
          },
          nb::arg("arg_types"), nb::kw_only(),
          nb::arg("arg_locs") = std::nullopt,
          "Creates and returns a new Block after this block "
          "(with given argument types and locations).")
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
           [](PyBlock &self, PyBlock &other) {
             return self.get().ptr == other.get().ptr;
           })
      .def("__eq__", [](PyBlock &self, nb::object &other) { return false; })
      .def("__hash__",
           [](PyBlock &self) {
             return static_cast<size_t>(llvm::hash_value(self.get().ptr));
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
          "Returns the assembly form of the block.")
      .def(
          "append",
          [](PyBlock &self, PyOperationBase &operation) {
            if (operation.getOperation().isAttached())
              operation.getOperation().detachFromParent();

            MlirOperation mlirOperation = operation.getOperation().get();
            mlirBlockAppendOwnedOperation(self.get(), mlirOperation);
            operation.getOperation().setAttached(
                self.getParentOperation().getObject());
          },
          nb::arg("operation"),
          "Appends an operation to this block. If the operation is currently "
          "in another block, it will be moved.");

  //----------------------------------------------------------------------------
  // Mapping of PyInsertionPoint.
  //----------------------------------------------------------------------------

  nb::class_<PyInsertionPoint>(m, "InsertionPoint")
      .def(nb::init<PyBlock &>(), nb::arg("block"),
           "Inserts after the last operation but still inside the block.")
      .def("__enter__", &PyInsertionPoint::contextEnter)
      .def("__exit__", &PyInsertionPoint::contextExit,
           nb::arg("exc_type").none(), nb::arg("exc_value").none(),
           nb::arg("traceback").none())
      .def_prop_ro_static(
          "current",
          [](nb::object & /*class*/) {
            auto *ip = PyThreadContextEntry::getDefaultInsertionPoint();
            if (!ip)
              throw nb::value_error("No current InsertionPoint");
            return ip;
          },
          "Gets the InsertionPoint bound to the current thread or raises "
          "ValueError if none has been set")
      .def(nb::init<PyOperationBase &>(), nb::arg("beforeOperation"),
           "Inserts before a referenced operation.")
      .def_static("at_block_begin", &PyInsertionPoint::atBlockBegin,
                  nb::arg("block"), "Inserts at the beginning of the block.")
      .def_static("at_block_terminator", &PyInsertionPoint::atBlockTerminator,
                  nb::arg("block"), "Inserts before the block terminator.")
      .def("insert", &PyInsertionPoint::insert, nb::arg("operation"),
           "Inserts an operation.")
      .def_prop_ro(
          "block", [](PyInsertionPoint &self) { return self.getBlock(); },
          "Returns the block that this InsertionPoint points to.")
      .def_prop_ro(
          "ref_operation",
          [](PyInsertionPoint &self) -> nb::object {
            auto refOperation = self.getRefOperation();
            if (refOperation)
              return refOperation->getObject();
            return nb::none();
          },
          "The reference operation before which new operations are "
          "inserted, or None if the insertion point is at the end of "
          "the block");

  //----------------------------------------------------------------------------
  // Mapping of PyAttribute.
  //----------------------------------------------------------------------------
  nb::class_<PyAttribute>(m, "Attribute")
      // Delegate to the PyAttribute copy constructor, which will also lifetime
      // extend the backing context which owns the MlirAttribute.
      .def(nb::init<PyAttribute &>(), nb::arg("cast_from_type"),
           "Casts the passed attribute to the generic Attribute")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyAttribute::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyAttribute::createFromCapsule)
      .def_static(
          "parse",
          [](const std::string &attrSpec, DefaultingPyMlirContext context) {
            PyMlirContext::ErrorCapture errors(context->getRef());
            MlirAttribute attr = mlirAttributeParseGet(
                context->get(), toMlirStringRef(attrSpec));
            if (mlirAttributeIsNull(attr))
              throw MLIRError("Unable to parse attribute", errors.take());
            return attr;
          },
          nb::arg("asm"), nb::arg("context").none() = nb::none(),
          "Parses an attribute from an assembly form. Raises an MLIRError on "
          "failure.")
      .def_prop_ro(
          "context",
          [](PyAttribute &self) { return self.getContext().getObject(); },
          "Context that owns the Attribute")
      .def_prop_ro("type",
                   [](PyAttribute &self) { return mlirAttributeGetType(self); })
      .def(
          "get_named",
          [](PyAttribute &self, std::string name) {
            return PyNamedAttribute(self, std::move(name));
          },
          nb::keep_alive<0, 1>(), "Binds a name to the attribute")
      .def("__eq__",
           [](PyAttribute &self, PyAttribute &other) { return self == other; })
      .def("__eq__", [](PyAttribute &self, nb::object &other) { return false; })
      .def("__hash__",
           [](PyAttribute &self) {
             return static_cast<size_t>(llvm::hash_value(self.get().ptr));
           })
      .def(
          "dump", [](PyAttribute &self) { mlirAttributeDump(self); },
          kDumpDocstring)
      .def(
          "__str__",
          [](PyAttribute &self) {
            PyPrintAccumulator printAccum;
            mlirAttributePrint(self, printAccum.getCallback(),
                               printAccum.getUserData());
            return printAccum.join();
          },
          "Returns the assembly form of the Attribute.")
      .def("__repr__",
           [](PyAttribute &self) {
             // Generally, assembly formats are not printed for __repr__ because
             // this can cause exceptionally long debug output and exceptions.
             // However, attribute values are generally considered useful and
             // are printed. This may need to be re-evaluated if debug dumps end
             // up being excessive.
             PyPrintAccumulator printAccum;
             printAccum.parts.append("Attribute(");
             mlirAttributePrint(self, printAccum.getCallback(),
                                printAccum.getUserData());
             printAccum.parts.append(")");
             return printAccum.join();
           })
      .def_prop_ro("typeid",
                   [](PyAttribute &self) -> MlirTypeID {
                     MlirTypeID mlirTypeID = mlirAttributeGetTypeID(self);
                     assert(!mlirTypeIDIsNull(mlirTypeID) &&
                            "mlirTypeID was expected to be non-null.");
                     return mlirTypeID;
                   })
      .def(MLIR_PYTHON_MAYBE_DOWNCAST_ATTR, [](PyAttribute &self) {
        MlirTypeID mlirTypeID = mlirAttributeGetTypeID(self);
        assert(!mlirTypeIDIsNull(mlirTypeID) &&
               "mlirTypeID was expected to be non-null.");
        std::optional<nb::callable> typeCaster =
            PyGlobals::get().lookupTypeCaster(mlirTypeID,
                                              mlirAttributeGetDialect(self));
        if (!typeCaster)
          return nb::cast(self);
        return typeCaster.value()(self);
      });

  //----------------------------------------------------------------------------
  // Mapping of PyNamedAttribute
  //----------------------------------------------------------------------------
  nb::class_<PyNamedAttribute>(m, "NamedAttribute")
      .def("__repr__",
           [](PyNamedAttribute &self) {
             PyPrintAccumulator printAccum;
             printAccum.parts.append("NamedAttribute(");
             printAccum.parts.append(
                 nb::str(mlirIdentifierStr(self.namedAttr.name).data,
                         mlirIdentifierStr(self.namedAttr.name).length));
             printAccum.parts.append("=");
             mlirAttributePrint(self.namedAttr.attribute,
                                printAccum.getCallback(),
                                printAccum.getUserData());
             printAccum.parts.append(")");
             return printAccum.join();
           })
      .def_prop_ro(
          "name",
          [](PyNamedAttribute &self) {
            return nb::str(mlirIdentifierStr(self.namedAttr.name).data,
                           mlirIdentifierStr(self.namedAttr.name).length);
          },
          "The name of the NamedAttribute binding")
      .def_prop_ro(
          "attr",
          [](PyNamedAttribute &self) { return self.namedAttr.attribute; },
          nb::keep_alive<0, 1>(),
          "The underlying generic attribute of the NamedAttribute binding");

  //----------------------------------------------------------------------------
  // Mapping of PyType.
  //----------------------------------------------------------------------------
  nb::class_<PyType>(m, "Type")
      // Delegate to the PyType copy constructor, which will also lifetime
      // extend the backing context which owns the MlirType.
      .def(nb::init<PyType &>(), nb::arg("cast_from_type"),
           "Casts the passed type to the generic Type")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyType::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyType::createFromCapsule)
      .def_static(
          "parse",
          [](std::string typeSpec, DefaultingPyMlirContext context) {
            PyMlirContext::ErrorCapture errors(context->getRef());
            MlirType type =
                mlirTypeParseGet(context->get(), toMlirStringRef(typeSpec));
            if (mlirTypeIsNull(type))
              throw MLIRError("Unable to parse type", errors.take());
            return type;
          },
          nb::arg("asm"), nb::arg("context").none() = nb::none(),
          kContextParseTypeDocstring)
      .def_prop_ro(
          "context", [](PyType &self) { return self.getContext().getObject(); },
          "Context that owns the Type")
      .def("__eq__", [](PyType &self, PyType &other) { return self == other; })
      .def(
          "__eq__", [](PyType &self, nb::object &other) { return false; },
          nb::arg("other").none())
      .def("__hash__",
           [](PyType &self) {
             return static_cast<size_t>(llvm::hash_value(self.get().ptr));
           })
      .def(
          "dump", [](PyType &self) { mlirTypeDump(self); }, kDumpDocstring)
      .def(
          "__str__",
          [](PyType &self) {
            PyPrintAccumulator printAccum;
            mlirTypePrint(self, printAccum.getCallback(),
                          printAccum.getUserData());
            return printAccum.join();
          },
          "Returns the assembly form of the type.")
      .def("__repr__",
           [](PyType &self) {
             // Generally, assembly formats are not printed for __repr__ because
             // this can cause exceptionally long debug output and exceptions.
             // However, types are an exception as they typically have compact
             // assembly forms and printing them is useful.
             PyPrintAccumulator printAccum;
             printAccum.parts.append("Type(");
             mlirTypePrint(self, printAccum.getCallback(),
                           printAccum.getUserData());
             printAccum.parts.append(")");
             return printAccum.join();
           })
      .def(MLIR_PYTHON_MAYBE_DOWNCAST_ATTR,
           [](PyType &self) {
             MlirTypeID mlirTypeID = mlirTypeGetTypeID(self);
             assert(!mlirTypeIDIsNull(mlirTypeID) &&
                    "mlirTypeID was expected to be non-null.");
             std::optional<nb::callable> typeCaster =
                 PyGlobals::get().lookupTypeCaster(mlirTypeID,
                                                   mlirTypeGetDialect(self));
             if (!typeCaster)
               return nb::cast(self);
             return typeCaster.value()(self);
           })
      .def_prop_ro("typeid", [](PyType &self) -> MlirTypeID {
        MlirTypeID mlirTypeID = mlirTypeGetTypeID(self);
        if (!mlirTypeIDIsNull(mlirTypeID))
          return mlirTypeID;
        auto origRepr = nb::cast<std::string>(nb::repr(nb::cast(self)));
        throw nb::value_error(
            (origRepr + llvm::Twine(" has no typeid.")).str().c_str());
      });

  //----------------------------------------------------------------------------
  // Mapping of PyTypeID.
  //----------------------------------------------------------------------------
  nb::class_<PyTypeID>(m, "TypeID")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyTypeID::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyTypeID::createFromCapsule)
      // Note, this tests whether the underlying TypeIDs are the same,
      // not whether the wrapper MlirTypeIDs are the same, nor whether
      // the Python objects are the same (i.e., PyTypeID is a value type).
      .def("__eq__",
           [](PyTypeID &self, PyTypeID &other) { return self == other; })
      .def("__eq__",
           [](PyTypeID &self, const nb::object &other) { return false; })
      // Note, this gives the hash value of the underlying TypeID, not the
      // hash value of the Python object, nor the hash value of the
      // MlirTypeID wrapper.
      .def("__hash__", [](PyTypeID &self) {
        return static_cast<size_t>(mlirTypeIDHashValue(self));
      });

  //----------------------------------------------------------------------------
  // Mapping of Value.
  //----------------------------------------------------------------------------
  nb::class_<PyValue>(m, "Value")
      .def(nb::init<PyValue &>(), nb::keep_alive<0, 1>(), nb::arg("value"))
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyValue::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyValue::createFromCapsule)
      .def_prop_ro(
          "context",
          [](PyValue &self) { return self.getParentOperation()->getContext(); },
          "Context in which the value lives.")
      .def(
          "dump", [](PyValue &self) { mlirValueDump(self.get()); },
          kDumpDocstring)
      .def_prop_ro(
          "owner",
          [](PyValue &self) -> nb::object {
            MlirValue v = self.get();
            if (mlirValueIsAOpResult(v)) {
              assert(
                  mlirOperationEqual(self.getParentOperation()->get(),
                                     mlirOpResultGetOwner(self.get())) &&
                  "expected the owner of the value in Python to match that in "
                  "the IR");
              return self.getParentOperation().getObject();
            }

            if (mlirValueIsABlockArgument(v)) {
              MlirBlock block = mlirBlockArgumentGetOwner(self.get());
              return nb::cast(PyBlock(self.getParentOperation(), block));
            }

            assert(false && "Value must be a block argument or an op result");
            return nb::none();
          })
      .def_prop_ro("uses",
                   [](PyValue &self) {
                     return PyOpOperandIterator(
                         mlirValueGetFirstUse(self.get()));
                   })
      .def("__eq__",
           [](PyValue &self, PyValue &other) {
             return self.get().ptr == other.get().ptr;
           })
      .def("__eq__", [](PyValue &self, nb::object other) { return false; })
      .def("__hash__",
           [](PyValue &self) {
             return static_cast<size_t>(llvm::hash_value(self.get().ptr));
           })
      .def(
          "__str__",
          [](PyValue &self) {
            PyPrintAccumulator printAccum;
            printAccum.parts.append("Value(");
            mlirValuePrint(self.get(), printAccum.getCallback(),
                           printAccum.getUserData());
            printAccum.parts.append(")");
            return printAccum.join();
          },
          kValueDunderStrDocstring)
      .def(
          "get_name",
          [](PyValue &self, bool useLocalScope) {
            PyPrintAccumulator printAccum;
            MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
            if (useLocalScope)
              mlirOpPrintingFlagsUseLocalScope(flags);
            MlirAsmState valueState =
                mlirAsmStateCreateForValue(self.get(), flags);
            mlirValuePrintAsOperand(self.get(), valueState,
                                    printAccum.getCallback(),
                                    printAccum.getUserData());
            mlirOpPrintingFlagsDestroy(flags);
            mlirAsmStateDestroy(valueState);
            return printAccum.join();
          },
          nb::arg("use_local_scope") = false)
      .def(
          "get_name",
          [](PyValue &self, PyAsmState &state) {
            PyPrintAccumulator printAccum;
            MlirAsmState valueState = state.get();
            mlirValuePrintAsOperand(self.get(), valueState,
                                    printAccum.getCallback(),
                                    printAccum.getUserData());
            return printAccum.join();
          },
          nb::arg("state"), kGetNameAsOperand)
      .def_prop_ro("type",
                   [](PyValue &self) { return mlirValueGetType(self.get()); })
      .def(
          "set_type",
          [](PyValue &self, const PyType &type) {
            return mlirValueSetType(self.get(), type);
          },
          nb::arg("type"))
      .def(
          "replace_all_uses_with",
          [](PyValue &self, PyValue &with) {
            mlirValueReplaceAllUsesOfWith(self.get(), with.get());
          },
          kValueReplaceAllUsesWithDocstring)
      .def(
          "replace_all_uses_except",
          [](MlirValue self, MlirValue with, PyOperation &exception) {
            MlirOperation exceptedUser = exception.get();
            mlirValueReplaceAllUsesExcept(self, with, 1, &exceptedUser);
          },
          nb::arg("with"), nb::arg("exceptions"),
          kValueReplaceAllUsesExceptDocstring)
      .def(
          "replace_all_uses_except",
          [](MlirValue self, MlirValue with, nb::list exceptions) {
            // Convert Python list to a SmallVector of MlirOperations
            llvm::SmallVector<MlirOperation> exceptionOps;
            for (nb::handle exception : exceptions) {
              exceptionOps.push_back(nb::cast<PyOperation &>(exception).get());
            }

            mlirValueReplaceAllUsesExcept(
                self, with, static_cast<intptr_t>(exceptionOps.size()),
                exceptionOps.data());
          },
          nb::arg("with"), nb::arg("exceptions"),
          kValueReplaceAllUsesExceptDocstring)
      .def(MLIR_PYTHON_MAYBE_DOWNCAST_ATTR,
           [](PyValue &self) { return self.maybeDownCast(); });
  PyBlockArgument::bind(m);
  PyOpResult::bind(m);
  PyOpOperand::bind(m);

  nb::class_<PyAsmState>(m, "AsmState")
      .def(nb::init<PyValue &, bool>(), nb::arg("value"),
           nb::arg("use_local_scope") = false)
      .def(nb::init<PyOperationBase &, bool>(), nb::arg("op"),
           nb::arg("use_local_scope") = false);

  //----------------------------------------------------------------------------
  // Mapping of SymbolTable.
  //----------------------------------------------------------------------------
  nb::class_<PySymbolTable>(m, "SymbolTable")
      .def(nb::init<PyOperationBase &>())
      .def("__getitem__", &PySymbolTable::dunderGetItem)
      .def("insert", &PySymbolTable::insert, nb::arg("operation"))
      .def("erase", &PySymbolTable::erase, nb::arg("operation"))
      .def("__delitem__", &PySymbolTable::dunderDel)
      .def("__contains__",
           [](PySymbolTable &table, const std::string &name) {
             return !mlirOperationIsNull(mlirSymbolTableLookup(
                 table, mlirStringRefCreate(name.data(), name.length())));
           })
      // Static helpers.
      .def_static("set_symbol_name", &PySymbolTable::setSymbolName,
                  nb::arg("symbol"), nb::arg("name"))
      .def_static("get_symbol_name", &PySymbolTable::getSymbolName,
                  nb::arg("symbol"))
      .def_static("get_visibility", &PySymbolTable::getVisibility,
                  nb::arg("symbol"))
      .def_static("set_visibility", &PySymbolTable::setVisibility,
                  nb::arg("symbol"), nb::arg("visibility"))
      .def_static("replace_all_symbol_uses",
                  &PySymbolTable::replaceAllSymbolUses, nb::arg("old_symbol"),
                  nb::arg("new_symbol"), nb::arg("from_op"))
      .def_static("walk_symbol_tables", &PySymbolTable::walkSymbolTables,
                  nb::arg("from_op"), nb::arg("all_sym_uses_visible"),
                  nb::arg("callback"));

  // Container bindings.
  PyBlockArgumentList::bind(m);
  PyBlockIterator::bind(m);
  PyBlockList::bind(m);
  PyOperationIterator::bind(m);
  PyOperationList::bind(m);
  PyOpAttributeMap::bind(m);
  PyOpOperandIterator::bind(m);
  PyOpOperandList::bind(m);
  PyOpResultList::bind(m);
  PyOpSuccessors::bind(m);
  PyRegionIterator::bind(m);
  PyRegionList::bind(m);

  // Debug bindings.
  PyGlobalDebugFlag::bind(m);

  // Attribute builder getter.
  PyAttrBuilderMap::bind(m);

  nb::register_exception_translator([](const std::exception_ptr &p,
                                       void *payload) {
    // We can't define exceptions with custom fields through pybind, so instead
    // the exception class is defined in python and imported here.
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const MLIRError &e) {
      nb::object obj = nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                           .attr("MLIRError")(e.message, e.errorDiagnostics);
      PyErr_SetObject(PyExc_Exception, obj.ptr());
    }
  });
}

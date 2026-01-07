//===- DialectTransform.cpp - 'transform' dialect submodule ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "IRInterfaces.h"
#include "Rewrite.h"
#include "mlir-c/Dialect/Transform.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "nanobind/nanobind.h"
#include <nanobind/trampoline.h>

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace transform {

//===----------------------------------------------------------------------===//
// TransformRewriter
//===----------------------------------------------------------------------===//
class PyTransformRewriter : public PyRewriterBase<PyTransformRewriter> {
public:
  static constexpr const char *pyClassName = "TransformRewriter";

  PyTransformRewriter(MlirTransformRewriter rewriter)
      : PyRewriterBase(mlirTransformRewriterAsBase(rewriter)) {}
};

//===----------------------------------------------------------------------===//
// TransformResults
//===----------------------------------------------------------------------===//
class PyTransformResults {
public:
  PyTransformResults(MlirTransformResults results) : results(results) {}

  MlirTransformResults get() const { return results; }

  void setOps(MlirValue result, const nanobind::list &ops) {
    std::vector<MlirOperation> opsVec;
    opsVec.reserve(ops.size());
    for (auto op : ops) {
      opsVec.push_back(nb::cast<MlirOperation>(op));
    }
    mlirTransformResultsSetOps(results, result, opsVec.size(), opsVec.data());
  }

  void setValues(MlirValue result, const nanobind::list &values) {
    std::vector<MlirValue> valuesVec;
    valuesVec.reserve(values.size());
    for (auto item : values) {
      valuesVec.push_back(nb::cast<MlirValue>(item));
    }
    mlirTransformResultsSetValues(results, result, valuesVec.size(),
                                  valuesVec.data());
  }

  void setParams(MlirValue result, const nanobind::list &params) {
    std::vector<MlirAttribute> paramsVec;
    paramsVec.reserve(params.size());
    for (auto item : params) {
      paramsVec.push_back(nb::cast<MlirAttribute>(item));
    }
    mlirTransformResultsSetParams(results, result, paramsVec.size(),
                                  paramsVec.data());
  }

  static void bind(nanobind::module_ &m) {
    nb::class_<PyTransformResults>(m, "TransformResults")
        .def(nb::init<MlirTransformResults>())
        .def("set_ops", &PyTransformResults::setOps,
             "Set the payload operations for a transform result.",
             nb::arg("result"), nb::arg("ops"))
        .def("set_values", &PyTransformResults::setValues,
             "Set the payload values for a transform result.",
             nb::arg("result"), nb::arg("values"))
        .def("set_params", &PyTransformResults::setParams,
             "Set the parameters for a transform result.", nb::arg("result"),
             nb::arg("params"));
  }

private:
  MlirTransformResults results;
};

//===----------------------------------------------------------------------===//
// TransformState
//===----------------------------------------------------------------------===//
class PyTransformState {
public:
  PyTransformState(MlirTransformState state) : state(state) {}

  MlirTransformState get() const { return state; }

  nanobind::list getPayloadOps(MlirValue value) {
    nanobind::list result;
    mlirTransformStateForEachPayloadOp(
        state, value,
        [](MlirOperation op, void *userData) {
          static_cast<nanobind::list *>(userData)->append(op);
        },
        &result);
    return result;
  }

  nanobind::list getPayloadValues(MlirValue value) {
    nanobind::list result;
    mlirTransformStateForEachPayloadValue(
        state, value,
        [](MlirValue val, void *userData) {
          static_cast<nanobind::list *>(userData)->append(val);
        },
        &result);
    return result;
  }

  nanobind::list getParams(MlirValue value) {
    nanobind::list result;
    mlirTransformStateForEachParam(
        state, value,
        [](MlirAttribute attr, void *userData) {
          static_cast<nanobind::list *>(userData)->append(attr);
        },
        &result);
    return result;
  }

  static void bind(nanobind::module_ &m) {
    nb::class_<PyTransformState>(m, "TransformState")
        .def(nb::init<MlirTransformState>())
        .def("get_payload_ops", &PyTransformState::getPayloadOps,
             "Get the payload operations associated with a transform IR value.",
             nb::arg("operand"))
        .def("get_payload_values", &PyTransformState::getPayloadValues,
             "Get the payload values associated with a transform IR value.",
             nb::arg("operand"))
        .def("get_params", &PyTransformState::getParams,
             "Get the parameters (attributes) associated with a transform IR "
             "value.",
             nb::arg("operand"));
  }

private:
  MlirTransformState state;
};

//===----------------------------------------------------------------------===//
// TransformOpInterface
//===----------------------------------------------------------------------===//
class PyTransformOpInterface
    : public PyConcreteOpInterface<PyTransformOpInterface> {
public:
  using PyConcreteOpInterface<PyTransformOpInterface>::PyConcreteOpInterface;

  constexpr static const char *pyClassName = "TransformOpInterface";
  constexpr static GetTypeIDFunctionTy getInterfaceID =
      &mlirTransformOpInterfaceTypeID;

  /// Attach a new TransformOpInterface FallbackModel to the named operation.
  /// The FallbackModel acts as a trampoline for callbacks on the Python class.
  static void attach(nb::object &pyClass, const std::string &opName,
                     DefaultingPyMlirContext ctx) {
    // Prepare the callbacks that will be used by the FallbackModel.
    MlirTransformOpInterfaceCallbacks callbacks;
    // Make the pointer to the Python class available to the callbacks.
    callbacks.userData = pyClass.ptr();
    nb::handle(static_cast<PyObject *>(callbacks.userData)).inc_ref();

    // The above ref bump is all we need as initialization, no need to run the
    // construct callback.
    callbacks.construct = nullptr;
    // Upon the FallbackModel's destruction, drop the ref to the Python class.
    callbacks.destruct = [](void *userData) {
      nb::handle(static_cast<PyObject *>(userData)).dec_ref();
    };
    // The apply callback which calls into Python.
    callbacks.apply = [](MlirOperation op, MlirTransformRewriter rewriter,
                         MlirTransformResults results, MlirTransformState state,
                         void *userData) -> MlirDiagnosedSilenceableFailure {
      nb::handle pyClass(static_cast<PyObject *>(userData));

      auto pyApply = nb::cast<nb::callable>(nb::getattr(pyClass, "apply"));

      auto pyRewriter = PyTransformRewriter(rewriter);
      auto pyResults = PyTransformResults(results);
      auto pyState = PyTransformState(state);

      // Invoke `pyClass.apply(op, rewriter, results, state)` as a classmethod.
      nb::object res = pyApply(op, pyRewriter, pyResults, pyState);

      return nb::cast<MlirDiagnosedSilenceableFailure>(res);
    };

    // The allows_repeated_handle_operands callback which calls into Python.
    callbacks.allowsRepeatedHandleOperands = [](MlirOperation op,
                                                void *userData) -> bool {
      nb::handle pyClass(static_cast<PyObject *>(userData));

      auto pyAllowRepeatedHandleOperands = nb::cast<nb::callable>(
          nb::getattr(pyClass, "allow_repeated_handle_operands"));

      // Invoke `pyClass.allow_repeated_handle_operands(op)` as a classmethod.
      nb::object res = pyAllowRepeatedHandleOperands(op);

      return nb::cast<bool>(res);
    };

    // Attach a FallbackModel, which calls into Python, to the named operation.
    mlirTransformOpInterfaceAttachFallbackModel(
        ctx->get(), wrap(StringRef(opName.c_str())), callbacks);
  }

  static void bindDerived(ClassTy &transformOpInterfaceClass) {
    transformOpInterfaceClass.attr("attach") =
        classmethod(&PyTransformOpInterface::attach, nb::arg("cls"),
                    nb::arg("op_name"), nb::arg("ctx") = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// AnyOpType
//===-------------------------------------------------------------------===//

struct AnyOpType : PyConcreteType<AnyOpType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATransformAnyOpType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTransformAnyOpTypeGetTypeID;
  static constexpr const char *pyClassName = "AnyOpType";
  static inline const MlirStringRef name = mlirTransformAnyOpTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return AnyOpType(context->getRef(),
                           mlirTransformAnyOpTypeGet(context.get()->get()));
        },
        "Get an instance of AnyOpType in the given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// AnyParamType
//===-------------------------------------------------------------------===//

struct AnyParamType : PyConcreteType<AnyParamType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATransformAnyParamType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTransformAnyParamTypeGetTypeID;
  static constexpr const char *pyClassName = "AnyParamType";
  static inline const MlirStringRef name = mlirTransformAnyParamTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return AnyParamType(context->getRef(), mlirTransformAnyParamTypeGet(
                                                     context.get()->get()));
        },
        "Get an instance of AnyParamType in the given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// AnyValueType
//===-------------------------------------------------------------------===//

struct AnyValueType : PyConcreteType<AnyValueType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATransformAnyValueType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTransformAnyValueTypeGetTypeID;
  static constexpr const char *pyClassName = "AnyValueType";
  static inline const MlirStringRef name = mlirTransformAnyValueTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return AnyValueType(context->getRef(), mlirTransformAnyValueTypeGet(
                                                     context.get()->get()));
        },
        "Get an instance of AnyValueType in the given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// OperationType
//===-------------------------------------------------------------------===//

struct OperationType : PyConcreteType<OperationType> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsATransformOperationType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTransformOperationTypeGetTypeID;
  static constexpr const char *pyClassName = "OperationType";
  static inline const MlirStringRef name = mlirTransformOperationTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &operationName, DefaultingPyMlirContext context) {
          MlirStringRef cOperationName =
              mlirStringRefCreate(operationName.data(), operationName.size());
          return OperationType(context->getRef(),
                               mlirTransformOperationTypeGet(
                                   context.get()->get(), cOperationName));
        },
        "Get an instance of OperationType for the given kind in the given "
        "context",
        nb::arg("operation_name"), nb::arg("context").none() = nb::none());
    c.def_prop_ro(
        "operation_name",
        [](const OperationType &type) {
          MlirStringRef operationName =
              mlirTransformOperationTypeGetOperationName(type);
          return nb::str(operationName.data, operationName.length);
        },
        "Get the name of the payload operation accepted by the handle.");
  }
};

//===-------------------------------------------------------------------===//
// ParamType
//===-------------------------------------------------------------------===//

struct ParamType : PyConcreteType<ParamType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATransformParamType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTransformParamTypeGetTypeID;
  static constexpr const char *pyClassName = "ParamType";
  static inline const MlirStringRef name = mlirTransformParamTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyType &type, DefaultingPyMlirContext context) {
          return ParamType(context->getRef(), mlirTransformParamTypeGet(
                                                  context.get()->get(), type));
        },
        "Get an instance of ParamType for the given type in the given context.",
        nb::arg("type"), nb::arg("context").none() = nb::none());
    c.def_prop_ro(
        "type",
        [](ParamType type) {
          return PyType(type.getContext(), mlirTransformParamTypeGetType(type))
              .maybeDownCast();
        },
        "Get the type this ParamType is associated with.");
  }
};

//===----------------------------------------------------------------------===//
// MemoryEffectsOpInterface helpers
//===----------------------------------------------------------------------===//

namespace {
void onlyReadsHandle(nb::list &operands, PyMemoryEffectsInstanceList effects) {
  std::vector<MlirOpOperand> operandsVec;
  operandsVec.reserve(operands.size());
  for (auto operand : operands)
    operandsVec.push_back(nb::cast<PyOpOperand>(operand));
  mlirTransformOnlyReadsHandle(operandsVec.data(), operandsVec.size(),
                               effects.effects);
};

void producesHandle(nb::list &results, PyMemoryEffectsInstanceList effects) {
  std::vector<MlirOpResult> resultsVec;
  resultsVec.reserve(results.size());
  for (auto result : results)
    resultsVec.push_back(nb::cast<PyOpResult>(result));
  mlirTransformProducesHandle(resultsVec.data(), resultsVec.size(),
                              effects.effects);
};
} // namespace

static void populateDialectTransformSubmodule(nb::module_ &m) {
  nb::enum_<MlirDiagnosedSilenceableFailure>(m, "DiagnosedSilenceableFailure")
      .value("Success", MlirDiagnosedSilenceableFailureSuccess)
      .value("SilenceableFailure",
             MlirDiagnosedSilenceableFailureSilenceableFailure)
      .value("DefiniteFailure", MlirDiagnosedSilenceableFailureDefiniteFailure);

  AnyOpType::bind(m);
  AnyParamType::bind(m);
  AnyValueType::bind(m);
  OperationType::bind(m);
  ParamType::bind(m);

  PyTransformRewriter::bind(m);
  PyTransformResults::bind(m);
  PyTransformState::bind(m);
  PyTransformOpInterface::bind(m);

  m.def("only_reads_handle", onlyReadsHandle,
        "Mark operands as only reading handles.", nb::arg("operands"),
        nb::arg("effects"));

  m.def("produces_handle", producesHandle, "Mark results as producing handles.",
        nb::arg("results"), nb::arg("effects"));
}
} // namespace transform
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsTransform, m) {
  m.doc() = "MLIR Transform dialect.";
  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::transform::
      populateDialectTransformSubmodule(m);
}

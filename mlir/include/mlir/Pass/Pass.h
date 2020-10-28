//===- Pass.h - Base classes for compiler passes ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PASS_H
#define MLIR_PASS_PASS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/Statistic.h"

namespace mlir {
namespace detail {
class OpToOpPassAdaptor;

/// The state for a single execution of a pass. This provides a unified
/// interface for accessing and initializing necessary state for pass execution.
struct PassExecutionState {
  PassExecutionState(Operation *ir, AnalysisManager analysisManager,
                     function_ref<LogicalResult(OpPassManager &, Operation *)>
                         pipelineExecutor)
      : irAndPassFailed(ir, false), analysisManager(analysisManager),
        pipelineExecutor(pipelineExecutor) {}

  /// The current operation being transformed and a bool for if the pass
  /// signaled a failure.
  llvm::PointerIntPair<Operation *, 1, bool> irAndPassFailed;

  /// The analysis manager for the operation.
  AnalysisManager analysisManager;

  /// The set of preserved analyses for the current execution.
  detail::PreservedAnalyses preservedAnalyses;

  /// This is a callback in the PassManager that allows to schedule dynamic
  /// pipelines that will be rooted at the provided operation.
  function_ref<LogicalResult(OpPassManager &, Operation *)> pipelineExecutor;
};
} // namespace detail

/// The abstract base pass class. This class contains information describing the
/// derived pass object, e.g its kind and abstract TypeID.
class Pass {
public:
  virtual ~Pass() = default;

  /// Returns the unique identifier that corresponds to this pass.
  TypeID getTypeID() const { return passID; }

  /// Returns the pass info for the specified pass class or null if unknown.
  static const PassInfo *lookupPassInfo(TypeID passID);
  template <typename PassT> static const PassInfo *lookupPassInfo() {
    return lookupPassInfo(TypeID::get<PassT>());
  }

  /// Returns the pass info for this pass.
  const PassInfo *lookupPassInfo() const { return lookupPassInfo(getTypeID()); }

  /// Returns the derived pass name.
  virtual StringRef getName() const = 0;

  /// Register dependent dialects for the current pass.
  /// A pass is expected to register the dialects it will create entities for
  /// (Operations, Types, Attributes), other than dialect that exists in the
  /// input. For example, a pass that converts from Linalg to Affine would
  /// register the Affine dialect but does not need to register Linalg.
  virtual void getDependentDialects(DialectRegistry &registry) const {}

  /// Returns the command line argument used when registering this pass. Return
  /// an empty string if one does not exist.
  virtual StringRef getArgument() const {
    if (const PassInfo *passInfo = lookupPassInfo())
      return passInfo->getPassArgument();
    return "";
  }

  /// Returns the name of the operation that this pass operates on, or None if
  /// this is a generic OperationPass.
  Optional<StringRef> getOpName() const { return opName; }

  //===--------------------------------------------------------------------===//
  // Options
  //===--------------------------------------------------------------------===//

  /// This class represents a specific pass option, with a provided data type.
  template <typename DataType,
            typename OptionParser = detail::PassOptions::OptionParser<DataType>>
  struct Option : public detail::PassOptions::Option<DataType, OptionParser> {
    template <typename... Args>
    Option(Pass &parent, StringRef arg, Args &&... args)
        : detail::PassOptions::Option<DataType, OptionParser>(
              parent.passOptions, arg, std::forward<Args>(args)...) {}
    using detail::PassOptions::Option<DataType, OptionParser>::operator=;
  };
  /// This class represents a specific pass option that contains a list of
  /// values of the provided data type.
  template <typename DataType,
            typename OptionParser = detail::PassOptions::OptionParser<DataType>>
  struct ListOption
      : public detail::PassOptions::ListOption<DataType, OptionParser> {
    template <typename... Args>
    ListOption(Pass &parent, StringRef arg, Args &&... args)
        : detail::PassOptions::ListOption<DataType, OptionParser>(
              parent.passOptions, arg, std::forward<Args>(args)...) {}
    using detail::PassOptions::ListOption<DataType, OptionParser>::operator=;
  };

  /// Attempt to initialize the options of this pass from the given string.
  LogicalResult initializeOptions(StringRef options);

  /// Prints out the pass in the textual representation of pipelines. If this is
  /// an adaptor pass, print with the op_name(sub_pass,...) format.
  void printAsTextualPipeline(raw_ostream &os, bool filterVerifier = true);

  //===--------------------------------------------------------------------===//
  // Statistics
  //===--------------------------------------------------------------------===//

  /// This class represents a single pass statistic. This statistic functions
  /// similarly to an unsigned integer value, and may be updated and incremented
  /// accordingly. This class can be used to provide additional information
  /// about the transformations and analyses performed by a pass.
  class Statistic : public llvm::Statistic {
  public:
    /// The statistic is initialized by the pass owner, a name, and a
    /// description.
    Statistic(Pass *owner, const char *name, const char *description);

    /// Assign the statistic to the given value.
    Statistic &operator=(unsigned value);

  private:
    /// Hide some of the details of llvm::Statistic that we don't use.
    using llvm::Statistic::getDebugType;
  };

  /// Returns the main statistics for this pass instance.
  ArrayRef<Statistic *> getStatistics() const { return statistics; }
  MutableArrayRef<Statistic *> getStatistics() { return statistics; }

protected:
  explicit Pass(TypeID passID, Optional<StringRef> opName = llvm::None)
      : passID(passID), opName(opName) {}
  Pass(const Pass &other) : Pass(other.passID, other.opName) {}

  /// Returns the current pass state.
  detail::PassExecutionState &getPassState() {
    assert(passState && "pass state was never initialized");
    return *passState;
  }

  /// Return the MLIR context for the current function being transformed.
  MLIRContext &getContext() { return *getOperation()->getContext(); }

  /// The polymorphic API that runs the pass over the currently held operation.
  virtual void runOnOperation() = 0;

  /// Schedule an arbitrary pass pipeline on the provided operation.
  /// This can be invoke any time in a pass to dynamic schedule more passes.
  /// The provided operation must be the current one or one nested below.
  LogicalResult runPipeline(OpPassManager &pipeline, Operation *op) {
    return passState->pipelineExecutor(pipeline, op);
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<Pass> clone() const {
    auto newInst = clonePass();
    newInst->copyOptionValuesFrom(this);
    return newInst;
  }

  /// Return the current operation being transformed.
  Operation *getOperation() {
    return getPassState().irAndPassFailed.getPointer();
  }

  /// Signal that some invariant was broken when running. The IR is allowed to
  /// be in an invalid state.
  void signalPassFailure() { getPassState().irAndPassFailed.setInt(true); }

  /// Query an analysis for the current ir unit.
  template <typename AnalysisT> AnalysisT &getAnalysis() {
    return getAnalysisManager().getAnalysis<AnalysisT>();
  }

  /// Query an analysis for the current ir unit of a specific derived operation
  /// type.
  template <typename AnalysisT, typename OpT>
  AnalysisT &getAnalysis() {
    return getAnalysisManager().getAnalysis<AnalysisT, OpT>();
  }

  /// Query a cached instance of an analysis for the current ir unit if one
  /// exists.
  template <typename AnalysisT>
  Optional<std::reference_wrapper<AnalysisT>> getCachedAnalysis() {
    return getAnalysisManager().getCachedAnalysis<AnalysisT>();
  }

  /// Mark all analyses as preserved.
  void markAllAnalysesPreserved() {
    getPassState().preservedAnalyses.preserveAll();
  }

  /// Mark the provided analyses as preserved.
  template <typename... AnalysesT> void markAnalysesPreserved() {
    getPassState().preservedAnalyses.preserve<AnalysesT...>();
  }
  void markAnalysesPreserved(TypeID id) {
    getPassState().preservedAnalyses.preserve(id);
  }

  /// Returns the analysis for the given parent operation if it exists.
  template <typename AnalysisT>
  Optional<std::reference_wrapper<AnalysisT>>
  getCachedParentAnalysis(Operation *parent) {
    return getAnalysisManager().getCachedParentAnalysis<AnalysisT>(parent);
  }

  /// Returns the analysis for the parent operation if it exists.
  template <typename AnalysisT>
  Optional<std::reference_wrapper<AnalysisT>> getCachedParentAnalysis() {
    return getAnalysisManager().getCachedParentAnalysis<AnalysisT>(
        getOperation()->getParentOp());
  }

  /// Returns the analysis for the given child operation if it exists.
  template <typename AnalysisT>
  Optional<std::reference_wrapper<AnalysisT>>
  getCachedChildAnalysis(Operation *child) {
    return getAnalysisManager().getCachedChildAnalysis<AnalysisT>(child);
  }

  /// Returns the analysis for the given child operation, or creates it if it
  /// doesn't exist.
  template <typename AnalysisT> AnalysisT &getChildAnalysis(Operation *child) {
    return getAnalysisManager().getChildAnalysis<AnalysisT>(child);
  }

  /// Returns the analysis for the given child operation of specific derived
  /// operation type, or creates it if it doesn't exist.
  template <typename AnalysisT, typename OpTy>
  AnalysisT &getChildAnalysis(OpTy child) {
    return getAnalysisManager().getChildAnalysis<AnalysisT>(child);
  }

  /// Returns the current analysis manager.
  AnalysisManager getAnalysisManager() {
    return getPassState().analysisManager;
  }

  /// Create a copy of this pass, ignoring statistics and options.
  virtual std::unique_ptr<Pass> clonePass() const = 0;

  /// Copy the option values from 'other', which is another instance of this
  /// pass.
  void copyOptionValuesFrom(const Pass *other);

private:

  /// Out of line virtual method to ensure vtables and metadata are emitted to a
  /// single .o file.
  virtual void anchor();

  /// Represents a unique identifier for the pass.
  TypeID passID;

  /// The name of the operation that this pass operates on, or None if this is a
  /// generic OperationPass.
  Optional<StringRef> opName;

  /// The current execution state for the pass.
  Optional<detail::PassExecutionState> passState;

  /// The set of statistics held by this pass.
  std::vector<Statistic *> statistics;

  /// The pass options registered to this pass instance.
  detail::PassOptions passOptions;

  /// Allow access to 'clone'.
  friend class OpPassManager;

  /// Allow access to 'passState'.
  friend detail::OpToOpPassAdaptor;

  /// Allow access to 'passOptions'.
  friend class PassInfo;
};

//===----------------------------------------------------------------------===//
// Pass Model Definitions
//===----------------------------------------------------------------------===//

/// Pass to transform an operation of a specific type.
///
/// Operation passes must not:
///   - modify any other operations within the parent region, as other threads
///     may be manipulating them concurrently.
///   - modify any state within the parent operation, this includes adding
///     additional operations.
///
/// Derived function passes are expected to provide the following:
///   - A 'void runOnOperation()' method.
///   - A 'StringRef getName() const' method.
///   - A 'std::unique_ptr<Pass> clonePass() const' method.
template <typename OpT = void> class OperationPass : public Pass {
protected:
  OperationPass(TypeID passID) : Pass(passID, OpT::getOperationName()) {}

  /// Support isa/dyn_cast functionality.
  static bool classof(const Pass *pass) {
    return pass->getOpName() == OpT::getOperationName();
  }

  /// Return the current operation being transformed.
  OpT getOperation() { return cast<OpT>(Pass::getOperation()); }

  /// Query an analysis for the current operation of the specific derived
  /// operation type.
  template <typename AnalysisT>
  AnalysisT &getAnalysis() {
    return Pass::getAnalysis<AnalysisT, OpT>();
  }
};

/// Pass to transform an operation.
///
/// Operation passes must not:
///   - modify any other operations within the parent region, as other threads
///     may be manipulating them concurrently.
///   - modify any state within the parent operation, this includes adding
///     additional operations.
///
/// Derived function passes are expected to provide the following:
///   - A 'void runOnOperation()' method.
///   - A 'StringRef getName() const' method.
///   - A 'std::unique_ptr<Pass> clonePass() const' method.
template <> class OperationPass<void> : public Pass {
protected:
  OperationPass(TypeID passID) : Pass(passID) {}
};

/// A model for providing function pass specific utilities.
///
/// Derived function passes are expected to provide the following:
///   - A 'void runOnFunction()' method.
///   - A 'StringRef getName() const' method.
///   - A 'std::unique_ptr<Pass> clonePass() const' method.
class FunctionPass : public OperationPass<FuncOp> {
public:
  using OperationPass<FuncOp>::OperationPass;

  /// The polymorphic API that runs the pass over the currently held function.
  virtual void runOnFunction() = 0;

  /// The polymorphic API that runs the pass over the currently held operation.
  void runOnOperation() final {
    if (!getFunction().isExternal())
      runOnFunction();
  }

  /// Return the current function being transformed.
  FuncOp getFunction() { return this->getOperation(); }
};

/// This class provides a CRTP wrapper around a base pass class to define
/// several necessary utility methods. This should only be used for passes that
/// are not suitably represented using the declarative pass specification(i.e.
/// tablegen backend).
template <typename PassT, typename BaseT> class PassWrapper : public BaseT {
public:
  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const Pass *pass) {
    return pass->getTypeID() == TypeID::get<PassT>();
  }

protected:
  PassWrapper() : BaseT(TypeID::get<PassT>()) {}

  /// Returns the derived pass name.
  StringRef getName() const override { return llvm::getTypeName<PassT>(); }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<PassT>(*static_cast<const PassT *>(this));
  }
};

} // end namespace mlir

#endif // MLIR_PASS_PASS_H

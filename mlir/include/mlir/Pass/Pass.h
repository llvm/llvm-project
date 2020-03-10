//===- Pass.h - Base classes for compiler passes ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PASS_H
#define MLIR_PASS_PASS_H

#include "mlir/IR/Function.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/Statistic.h"

namespace mlir {
namespace detail {
/// The state for a single execution of a pass. This provides a unified
/// interface for accessing and initializing necessary state for pass execution.
struct PassExecutionState {
  PassExecutionState(Operation *ir, AnalysisManager analysisManager)
      : irAndPassFailed(ir, false), analysisManager(analysisManager) {}

  /// The current operation being transformed and a bool for if the pass
  /// signaled a failure.
  llvm::PointerIntPair<Operation *, 1, bool> irAndPassFailed;

  /// The analysis manager for the operation.
  AnalysisManager analysisManager;

  /// The set of preserved analyses for the current execution.
  detail::PreservedAnalyses preservedAnalyses;
};
} // namespace detail

/// The abstract base pass class. This class contains information describing the
/// derived pass object, e.g its kind and abstract PassInfo.
class Pass {
public:
  virtual ~Pass() = default;

  /// Returns the unique identifier that corresponds to this pass.
  const PassID *getPassID() const { return passID; }

  /// Returns the pass info for the specified pass class or null if unknown.
  static const PassInfo *lookupPassInfo(const PassID *passID);
  template <typename PassT> static const PassInfo *lookupPassInfo() {
    return lookupPassInfo(PassID::getID<PassT>());
  }

  /// Returns the pass info for this pass.
  const PassInfo *lookupPassInfo() const { return lookupPassInfo(getPassID()); }

  /// Returns the derived pass name.
  virtual StringRef getName() = 0;

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
  void printAsTextualPipeline(raw_ostream &os);

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
  explicit Pass(const PassID *passID, Optional<StringRef> opName = llvm::None)
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

  /// A clone method to create a copy of this pass.
  virtual std::unique_ptr<Pass> clone() const = 0;

  /// Return the current operation being transformed.
  Operation *getOperation() {
    return getPassState().irAndPassFailed.getPointer();
  }

  /// Returns the current analysis manager.
  AnalysisManager getAnalysisManager() {
    return getPassState().analysisManager;
  }

  /// Copy the option values from 'other', which is another instance of this
  /// pass.
  void copyOptionValuesFrom(const Pass *other);

private:
  /// Forwarding function to execute this pass on the given operation.
  LLVM_NODISCARD
  LogicalResult run(Operation *op, AnalysisManager am);

  /// Out of line virtual method to ensure vtables and metadata are emitted to a
  /// single .o file.
  virtual void anchor();

  /// Represents a unique identifier for the pass.
  const PassID *passID;

  /// The name of the operation that this pass operates on, or None if this is a
  /// generic OperationPass.
  Optional<StringRef> opName;

  /// The current execution state for the pass.
  Optional<detail::PassExecutionState> passState;

  /// The set of statistics held by this pass.
  std::vector<Statistic *> statistics;

  /// The pass options registered to this pass instance.
  detail::PassOptions passOptions;

  /// Allow access to 'clone' and 'run'.
  friend class OpPassManager;

  /// Allow access to 'passOptions'.
  friend class PassInfo;
};

//===----------------------------------------------------------------------===//
// Pass Model Definitions
//===----------------------------------------------------------------------===//
namespace detail {
/// The opaque CRTP model of a pass. This class provides utilities for derived
/// pass execution and handles all of the necessary polymorphic API.
template <typename PassT, typename BasePassT>
class PassModel : public BasePassT {
public:
  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const Pass *pass) {
    return pass->getPassID() == PassID::getID<PassT>();
  }

protected:
  explicit PassModel(Optional<StringRef> opName = llvm::None)
      : BasePassT(PassID::getID<PassT>(), opName) {}

  /// Signal that some invariant was broken when running. The IR is allowed to
  /// be in an invalid state.
  void signalPassFailure() {
    this->getPassState().irAndPassFailed.setInt(true);
  }

  /// Query an analysis for the current ir unit.
  template <typename AnalysisT> AnalysisT &getAnalysis() {
    return this->getAnalysisManager().template getAnalysis<AnalysisT>();
  }

  /// Query a cached instance of an analysis for the current ir unit if one
  /// exists.
  template <typename AnalysisT>
  Optional<std::reference_wrapper<AnalysisT>> getCachedAnalysis() {
    return this->getAnalysisManager().template getCachedAnalysis<AnalysisT>();
  }

  /// Mark all analyses as preserved.
  void markAllAnalysesPreserved() {
    this->getPassState().preservedAnalyses.preserveAll();
  }

  /// Mark the provided analyses as preserved.
  template <typename... AnalysesT> void markAnalysesPreserved() {
    this->getPassState().preservedAnalyses.template preserve<AnalysesT...>();
  }
  void markAnalysesPreserved(const AnalysisID *id) {
    this->getPassState().preservedAnalyses.preserve(id);
  }

  /// Returns the derived pass name.
  StringRef getName() override {
    StringRef name = llvm::getTypeName<PassT>();
    if (!name.consume_front("mlir::"))
      name.consume_front("(anonymous namespace)::");
    return name;
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<Pass> clone() const override {
    auto newInst = std::make_unique<PassT>(*static_cast<const PassT *>(this));
    newInst->copyOptionValuesFrom(this);
    return newInst;
  }

  /// Returns the analysis for the parent operation if it exists.
  template <typename AnalysisT>
  Optional<std::reference_wrapper<AnalysisT>>
  getCachedParentAnalysis(Operation *parent) {
    return this->getAnalysisManager()
        .template getCachedParentAnalysis<AnalysisT>(parent);
  }
  template <typename AnalysisT>
  Optional<std::reference_wrapper<AnalysisT>> getCachedParentAnalysis() {
    return this->getAnalysisManager()
        .template getCachedParentAnalysis<AnalysisT>(
            this->getOperation()->getParentOp());
  }

  /// Returns the analysis for the given child operation if it exists.
  template <typename AnalysisT>
  Optional<std::reference_wrapper<AnalysisT>>
  getCachedChildAnalysis(Operation *child) {
    return this->getAnalysisManager()
        .template getCachedChildAnalysis<AnalysisT>(child);
  }

  /// Returns the analysis for the given child operation, or creates it if it
  /// doesn't exist.
  template <typename AnalysisT> AnalysisT &getChildAnalysis(Operation *child) {
    return this->getAnalysisManager().template getChildAnalysis<AnalysisT>(
        child);
  }
};
} // end namespace detail

/// Utility base class for OpPass below to denote an opaque pass operating on a
/// specific operation type.
template <typename OpT> class OpPassBase : public Pass {
public:
  using Pass::Pass;

  /// Support isa/dyn_cast functionality.
  static bool classof(const Pass *pass) {
    return pass->getOpName() == OpT::getOperationName();
  }
};

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
template <typename PassT, typename OpT = void>
class OperationPass : public detail::PassModel<PassT, OpPassBase<OpT>> {
protected:
  OperationPass()
      : detail::PassModel<PassT, OpPassBase<OpT>>(OpT::getOperationName()) {}

  /// Return the current operation being transformed.
  OpT getOperation() { return cast<OpT>(Pass::getOperation()); }
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
template <typename PassT>
struct OperationPass<PassT, void> : public detail::PassModel<PassT, Pass> {};

/// A model for providing function pass specific utilities.
///
/// Derived function passes are expected to provide the following:
///   - A 'void runOnFunction()' method.
template <typename T> struct FunctionPass : public OperationPass<T, FuncOp> {
  /// The polymorphic API that runs the pass over the currently held function.
  virtual void runOnFunction() = 0;

  /// The polymorphic API that runs the pass over the currently held operation.
  void runOnOperation() final {
    if (!getFunction().isExternal())
      runOnFunction();
  }

  /// Return the current module being transformed.
  FuncOp getFunction() { return this->getOperation(); }
};

/// A model for providing module pass specific utilities.
///
/// Derived module passes are expected to provide the following:
///   - A 'void runOnModule()' method.
template <typename T> struct ModulePass : public OperationPass<T, ModuleOp> {
  /// The polymorphic API that runs the pass over the currently held module.
  virtual void runOnModule() = 0;

  /// The polymorphic API that runs the pass over the currently held operation.
  void runOnOperation() final { runOnModule(); }

  /// Return the current module being transformed.
  ModuleOp getModule() { return this->getOperation(); }
};
} // end namespace mlir

#endif // MLIR_PASS_PASS_H

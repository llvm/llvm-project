//===- PassManager internal APIs and implementation details -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header provides internal APIs and implementation details used by the
/// pass management interfaces exposed in PassManager.h. To understand more
/// context of why these particular interfaces are needed, see that header
/// file. None of these APIs should be used elsewhere.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PASSMANAGERINTERNAL_H
#define LLVM_IR_PASSMANAGERINTERNAL_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Analysis.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <type_traits>
#include <utility>

namespace llvm {

template <typename IRUnitT> class AllAnalysesOn;
template <typename IRUnitT, typename... ExtraArgTs> class AnalysisManager;
class PreservedAnalyses;

// Implementation details of the pass manager interfaces.
namespace detail {

/// Template for the abstract base class used to dispatch over pass objects.
/// This doesn't use virtual functions to avoid vtables, which cost a fair
/// amount of storage that needs to be relocated in PIC builds and add an extra
/// indirection on dispatch.
template <typename IRUnitT, typename AnalysisManagerT, typename... ExtraArgTs>
class PassConcept {
private:
  using DestructorTy = void (*)(PassConcept &);
  using RunTy = PreservedAnalyses (*)(PassConcept &, IRUnitT &,
                                      AnalysisManagerT &, ExtraArgTs...);
  using PrintPipelineTy =
      void (*)(PassConcept &, raw_ostream &,
               function_ref<StringRef(StringRef)> MapClassName2PassName);

  StringRef Name;
  bool IsRequired;

  DestructorTy Destructor;
  RunTy Run;
  PrintPipelineTy PrintPipeline;

protected:
  PassConcept(StringRef Name, bool IsRequired, DestructorTy Destructor,
              RunTy Run, PrintPipelineTy PrintPipeline)
      : Name(Name), IsRequired(IsRequired), Destructor(Destructor), Run(Run),
        PrintPipeline(PrintPipeline) {}

public:
  // Boiler plate necessary for the container of derived classes.
  ~PassConcept() { Destructor(*this); }

  // Passes are immovable.
  PassConcept(const PassConcept &) = delete;
  PassConcept &operator=(const PassConcept &) = delete;

  /// Run the pass.
  PreservedAnalyses run(IRUnitT &IR, AnalysisManagerT &AM,
                        ExtraArgTs... ExtraArgs) {
    return Run(*this, IR, AM, std::forward<ExtraArgTs>(ExtraArgs)...);
  }

  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName) {
    PrintPipeline(*this, OS, MapClassName2PassName);
  }

  /// Get name of a pass.
  StringRef name() const { return Name; }

  /// Indicate whether a pass can optionally be exempted from skipping by
  /// PassInstrumentation.
  /// To opt-in, pass should implement `static bool isRequired()`, or inherit
  /// from `RequiredPassInfoMixin` or `OptionalPassInfoMixin`.
  /// It's no-op to have `isRequired` always return false since that is the
  /// default.
  bool isRequired() const { return IsRequired; }
};

/// A template wrapper used to implement PassConcept.
///
/// Can be instantiated for any object which provides a \c run method accepting
/// an \c IRUnitT& and an \c AnalysisManager<IRUnit>&.
template <typename IRUnitT, typename PassT, typename AnalysisManagerT,
          typename... ExtraArgTs>
class PassModel final
    : public PassConcept<IRUnitT, AnalysisManagerT, ExtraArgTs...> {
private:
  using PassConceptT = PassConcept<IRUnitT, AnalysisManagerT, ExtraArgTs...>;

  /// Storage for PassT. We don't use a PassT here, because the destructor of
  /// PassModel will not be called -- the PassT instance has to be destructed
  /// in destructorImpl.
  alignas(PassT) char PassBytes[sizeof(PassT)];

  static PassT &getPass(PassConceptT &Self) {
    return *reinterpret_cast<PassT *>(static_cast<PassModel &>(Self).PassBytes);
  }

  static void destructorImpl(PassConceptT &Self) { getPass(Self).~PassT(); }

  static PreservedAnalyses runImpl(PassConceptT &Self, IRUnitT &IR,
                                   AnalysisManagerT &AM,
                                   ExtraArgTs... ExtraArgs) {
    return getPass(Self).run(IR, AM, ExtraArgs...);
  }

  static void
  printPipelineImpl(PassConceptT &Self, raw_ostream &OS,
                    function_ref<StringRef(StringRef)> MapClassName2PassName) {
    getPass(Self).printPipeline(OS, MapClassName2PassName);
  }

public:
  explicit PassModel(PassT Pass)
      : PassConceptT(PassT::name(), PassT::isRequired(), destructorImpl,
                     runImpl, printPipelineImpl) {
    new (PassBytes) PassT(std::move(Pass));
  }
};

/// Abstract concept of an analysis result.
///
/// This concept is parameterized over the IR unit that this result pertains
/// to.
template <typename IRUnitT, typename InvalidatorT>
struct AnalysisResultConcept {
  virtual ~AnalysisResultConcept() = default;

  /// Method to try and mark a result as invalid.
  ///
  /// When the outer analysis manager detects a change in some underlying
  /// unit of the IR, it will call this method on all of the results cached.
  ///
  /// \p PA is a set of preserved analyses which can be used to avoid
  /// invalidation because the pass which changed the underlying IR took care
  /// to update or preserve the analysis result in some way.
  ///
  /// \p Inv is typically a \c AnalysisManager::Invalidator object that can be
  /// used by a particular analysis result to discover if other analyses
  /// results are also invalidated in the event that this result depends on
  /// them. See the documentation in the \c AnalysisManager for more details.
  ///
  /// \returns true if the result is indeed invalid (the default).
  virtual bool invalidate(IRUnitT &IR, const PreservedAnalyses &PA,
                          InvalidatorT &Inv) = 0;
};

/// SFINAE metafunction for computing whether \c ResultT provides an
/// \c invalidate member function.
template <typename IRUnitT, typename ResultT> class ResultHasInvalidateMethod {
  using EnabledType = char;
  struct DisabledType {
    char a, b;
  };

  // Purely to help out MSVC which fails to disable the below specialization,
  // explicitly enable using the result type's invalidate routine if we can
  // successfully call that routine.
  template <typename T> struct Nonce { using Type = EnabledType; };
  template <typename T>
  static typename Nonce<decltype(std::declval<T>().invalidate(
      std::declval<IRUnitT &>(), std::declval<PreservedAnalyses>()))>::Type
      check(rank<2>);

  // First we define an overload that can only be taken if there is no
  // invalidate member. We do this by taking the address of an invalidate
  // member in an adjacent base class of a derived class. This would be
  // ambiguous if there were an invalidate member in the result type.
  template <typename T, typename U> static DisabledType NonceFunction(T U::*);
  struct CheckerBase { int invalidate; };
  template <typename T> struct Checker : CheckerBase, std::remove_cv_t<T> {};
  template <typename T>
  static decltype(NonceFunction(&Checker<T>::invalidate)) check(rank<1>);

  // Now we have the fallback that will only be reached when there is an
  // invalidate member, and enables the trait.
  template <typename T>
  static EnabledType check(rank<0>);

public:
  enum { Value = sizeof(check<ResultT>(rank<2>())) == sizeof(EnabledType) };
};

/// Wrapper to model the analysis result concept.
///
/// By default, this will implement the invalidate method with a trivial
/// implementation so that the actual analysis result doesn't need to provide
/// an invalidation handler. It is only selected when the invalidation handler
/// is not part of the ResultT's interface.
template <typename IRUnitT, typename PassT, typename ResultT,
          typename InvalidatorT,
          bool HasInvalidateHandler =
              ResultHasInvalidateMethod<IRUnitT, ResultT>::Value>
struct AnalysisResultModel;

/// Specialization of \c AnalysisResultModel which provides the default
/// invalidate functionality.
template <typename IRUnitT, typename PassT, typename ResultT,
          typename InvalidatorT>
struct AnalysisResultModel<IRUnitT, PassT, ResultT, InvalidatorT, false>
    : AnalysisResultConcept<IRUnitT, InvalidatorT> {
  explicit AnalysisResultModel(ResultT Result) : Result(std::move(Result)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisResultModel(const AnalysisResultModel &Arg) : Result(Arg.Result) {}
  AnalysisResultModel(AnalysisResultModel &&Arg)
      : Result(std::move(Arg.Result)) {}

  friend void swap(AnalysisResultModel &LHS, AnalysisResultModel &RHS) {
    using std::swap;
    swap(LHS.Result, RHS.Result);
  }

  AnalysisResultModel &operator=(AnalysisResultModel RHS) {
    swap(*this, RHS);
    return *this;
  }

  /// The model bases invalidation solely on being in the preserved set.
  //
  // FIXME: We should actually use two different concepts for analysis results
  // rather than two different models, and avoid the indirect function call for
  // ones that use the trivial behavior.
  bool invalidate(IRUnitT &, const PreservedAnalyses &PA,
                  InvalidatorT &) override {
    auto PAC = PA.template getChecker<PassT>();
    return !PAC.preserved() &&
           !PAC.template preservedSet<AllAnalysesOn<IRUnitT>>();
  }

  ResultT Result;
};

/// Specialization of \c AnalysisResultModel which delegates invalidate
/// handling to \c ResultT.
template <typename IRUnitT, typename PassT, typename ResultT,
          typename InvalidatorT>
struct AnalysisResultModel<IRUnitT, PassT, ResultT, InvalidatorT, true>
    : AnalysisResultConcept<IRUnitT, InvalidatorT> {
  explicit AnalysisResultModel(ResultT Result) : Result(std::move(Result)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisResultModel(const AnalysisResultModel &Arg) : Result(Arg.Result) {}
  AnalysisResultModel(AnalysisResultModel &&Arg)
      : Result(std::move(Arg.Result)) {}

  friend void swap(AnalysisResultModel &LHS, AnalysisResultModel &RHS) {
    using std::swap;
    swap(LHS.Result, RHS.Result);
  }

  AnalysisResultModel &operator=(AnalysisResultModel RHS) {
    swap(*this, RHS);
    return *this;
  }

  /// The model delegates to the \c ResultT method.
  bool invalidate(IRUnitT &IR, const PreservedAnalyses &PA,
                  InvalidatorT &Inv) override {
    return Result.invalidate(IR, PA, Inv);
  }

  ResultT Result;
};

/// Abstract concept of an analysis pass.
///
/// This concept is parameterized over the IR unit that it can run over and
/// produce an analysis result.
template <typename IRUnitT, typename InvalidatorT, typename... ExtraArgTs>
struct AnalysisPassConcept {
  virtual ~AnalysisPassConcept() = default;

  /// Method to run this analysis over a unit of IR.
  /// \returns A unique_ptr to the analysis result object to be queried by
  /// users.
  virtual std::unique_ptr<AnalysisResultConcept<IRUnitT, InvalidatorT>>
  run(IRUnitT &IR, AnalysisManager<IRUnitT, ExtraArgTs...> &AM,
      ExtraArgTs... ExtraArgs) = 0;

  /// Polymorphic method to access the name of a pass.
  virtual StringRef name() const = 0;
};

/// Wrapper to model the analysis pass concept.
///
/// Can wrap any type which implements a suitable \c run method. The method
/// must accept an \c IRUnitT& and an \c AnalysisManager<IRUnitT>& as arguments
/// and produce an object which can be wrapped in a \c AnalysisResultModel.
template <typename IRUnitT, typename PassT, typename InvalidatorT,
          typename... ExtraArgTs>
struct AnalysisPassModel
    : AnalysisPassConcept<IRUnitT, InvalidatorT, ExtraArgTs...> {
  explicit AnalysisPassModel(PassT Pass) : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisPassModel(const AnalysisPassModel &Arg) : Pass(Arg.Pass) {}
  AnalysisPassModel(AnalysisPassModel &&Arg) : Pass(std::move(Arg.Pass)) {}

  friend void swap(AnalysisPassModel &LHS, AnalysisPassModel &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
  }

  AnalysisPassModel &operator=(AnalysisPassModel RHS) {
    swap(*this, RHS);
    return *this;
  }

  // FIXME: Replace PassT::Result with type traits when we use C++11.
  using ResultModelT =
      AnalysisResultModel<IRUnitT, PassT, typename PassT::Result, InvalidatorT>;

  /// The model delegates to the \c PassT::run method.
  ///
  /// The return is wrapped in an \c AnalysisResultModel.
  std::unique_ptr<AnalysisResultConcept<IRUnitT, InvalidatorT>>
  run(IRUnitT &IR, AnalysisManager<IRUnitT, ExtraArgTs...> &AM,
      ExtraArgTs... ExtraArgs) override {
    return std::make_unique<ResultModelT>(
        Pass.run(IR, AM, std::forward<ExtraArgTs>(ExtraArgs)...));
  }

  /// The model delegates to a static \c PassT::name method.
  ///
  /// The returned string ref must point to constant immutable data!
  StringRef name() const override { return PassT::name(); }

  PassT Pass;
};

} // end namespace detail

} // end namespace llvm

#endif // LLVM_IR_PASSMANAGERINTERNAL_H

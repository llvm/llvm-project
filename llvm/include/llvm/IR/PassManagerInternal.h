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
  using DestroyTy = void (*)(PassConcept &);
  using RunTy = PreservedAnalyses (*)(PassConcept &, IRUnitT &,
                                      AnalysisManagerT &, ExtraArgTs...);
  using PrintPipelineTy =
      void (*)(PassConcept &, raw_ostream &,
               function_ref<StringRef(StringRef)> MapClassName2PassName);

public:
  struct Deleter {
    void operator()(PassConcept *P) { P->Destroy(*P); }
  };

  using unique_ptr = std::unique_ptr<PassConcept, Deleter>;

private:
  StringRef Name;
  bool IsRequired;

  DestroyTy Destroy;
  RunTy Run;
  PrintPipelineTy PrintPipeline;

protected:
  PassConcept(StringRef Name, bool IsRequired, DestroyTy Destroy, RunTy Run,
              PrintPipelineTy PrintPipeline)
      : Name(Name), IsRequired(IsRequired), Destroy(Destroy), Run(Run),
        PrintPipeline(PrintPipeline) {}

  // Note: this is intentionally not public to catch uses of delete and
  // unique_ptr<PassConcept>.
  void operator delete(void *P) { ::operator delete(P); }

public:
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

  PassT Pass;

  static PassT &getPass(PassConceptT &Self) {
    return static_cast<PassModel &>(Self).Pass;
  }

  static void destroyImpl(PassConceptT &Self) {
    delete static_cast<PassModel *>(&Self);
  }

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

  explicit PassModel(PassT &&Pass)
      : PassConceptT(PassT::name(), PassT::isRequired(), destroyImpl, runImpl,
                     printPipelineImpl),
        Pass(std::move(Pass)) {}

public:
  static typename PassConceptT::unique_ptr create(PassT &&Pass) {
    return typename PassConceptT::unique_ptr(new PassModel(std::move(Pass)));
  }
};

/// Abstract concept of an analysis result.
///
/// This concept is parameterized over the IR unit that this result pertains
/// to.
template <typename IRUnitT, typename InvalidatorT>
struct AnalysisResultConcept {
private:
  using DestroyTy = void (*)(AnalysisResultConcept &);
  using InvalidateTy = bool (*)(AnalysisResultConcept &, IRUnitT &,
                                const PreservedAnalyses &, InvalidatorT &);

public:
  struct Deleter {
    void operator()(AnalysisResultConcept *C) { C->Destroy(*C); }
  };

  using unique_ptr = std::unique_ptr<AnalysisResultConcept, Deleter>;

protected:
  DestroyTy Destroy;
  InvalidateTy Invalidate = nullptr;

  AnalysisResultConcept(DestroyTy Destroy) : Destroy(Destroy) {}

public:
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
  bool invalidate(IRUnitT &IR, const PreservedAnalyses &PA, InvalidatorT &Inv) {
    return Invalidate(*this, IR, PA, Inv);
  }
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
template <typename IRUnitT, typename PassT, typename ResultT,
          typename InvalidatorT>
struct AnalysisResultModel
    : public AnalysisResultConcept<IRUnitT, InvalidatorT> {
  using AnalysisResultConceptT = AnalysisResultConcept<IRUnitT, InvalidatorT>;

  ResultT Result;

  static void destroyImpl(AnalysisResultConceptT &Self) {
    delete static_cast<AnalysisResultModel *>(&Self);
  }

  template <typename... ExtraArgTs>
  AnalysisResultModel(PassT &Pass, IRUnitT &IR,
                      AnalysisManager<IRUnitT, ExtraArgTs...> &AM,
                      ExtraArgTs &&...ExtraArgs)
      : AnalysisResultConceptT(destroyImpl),
        Result(Pass.run(IR, AM, std::forward<ExtraArgTs>(ExtraArgs)...)) {
    this->Invalidate = [](AnalysisResultConceptT &Self, IRUnitT &IR,
                          const PreservedAnalyses &PA, InvalidatorT &Inv) {
      if constexpr (ResultHasInvalidateMethod<IRUnitT, ResultT>::Value) {
        ResultT &Result = static_cast<AnalysisResultModel &>(Self).Result;
        return Result.invalidate(IR, PA, Inv);
      }
      auto PAC = PA.template getChecker<PassT>();
      return !PAC.preserved() &&
             !PAC.template preservedSet<AllAnalysesOn<IRUnitT>>();
    };
  }
};

/// Abstract concept of an analysis pass.
///
/// This concept is parameterized over the IR unit that it can run over and
/// produce an analysis result.
template <typename IRUnitT, typename InvalidatorT, typename... ExtraArgTs>
class AnalysisPassConcept {
public:
  using ResultPtrT =
      typename AnalysisResultConcept<IRUnitT, InvalidatorT>::unique_ptr;
  using AnalysisManagerT = AnalysisManager<IRUnitT, ExtraArgTs...>;

  struct Deleter {
    void operator()(AnalysisPassConcept *P) { P->Destroy(*P); }
  };
  using unique_ptr = std::unique_ptr<AnalysisPassConcept, Deleter>;

private:
  using DestroyTy = void (*)(AnalysisPassConcept &);
  using RunTy = ResultPtrT (*)(AnalysisPassConcept &, IRUnitT &,
                               AnalysisManagerT &, ExtraArgTs &&...);

  StringRef Name;

  DestroyTy Destroy;
  RunTy Run;

protected:
  AnalysisPassConcept(StringRef Name, DestroyTy Destroy, RunTy Run)
      : Name(Name), Destroy(Destroy), Run(Run) {}

public:
  // Passes are immovable.
  AnalysisPassConcept(const AnalysisPassConcept &) = delete;
  AnalysisPassConcept &operator=(const AnalysisPassConcept &) = delete;

  /// Method to run this analysis over a unit of IR.
  /// \returns A unique_ptr to the analysis result object to be queried by
  /// users.
  ResultPtrT run(IRUnitT &IR, AnalysisManagerT &AM, ExtraArgTs &&...ExtraArgs) {
    return Run(*this, IR, AM, std::forward<ExtraArgTs>(ExtraArgs)...);
  }

  /// Get the name of the pass.
  StringRef name() const { return Name; }
};

/// Wrapper to model the analysis pass concept.
///
/// Can wrap any type which implements a suitable \c run method. The method
/// must accept an \c IRUnitT& and an \c AnalysisManager<IRUnitT>& as arguments
/// and produce an object which can be wrapped in a \c AnalysisResultModel.
template <typename IRUnitT, typename PassT, typename InvalidatorT,
          typename... ExtraArgTs>
class AnalysisPassModel final
    : public AnalysisPassConcept<IRUnitT, InvalidatorT, ExtraArgTs...> {
  using AnalysisPassConceptT =
      AnalysisPassConcept<IRUnitT, InvalidatorT, ExtraArgTs...>;
  using ResultModelT =
      AnalysisResultModel<IRUnitT, PassT, typename PassT::Result, InvalidatorT>;

  PassT Pass;

  static void destroyImpl(AnalysisPassConceptT &Self) {
    delete static_cast<AnalysisPassModel *>(&Self);
  }

  static typename ResultModelT::unique_ptr
  runImpl(AnalysisPassConceptT &Self, IRUnitT &IR,
          AnalysisManager<IRUnitT, ExtraArgTs...> &AM,
          ExtraArgTs &&...ExtraArgs) {
    PassT &Pass = static_cast<AnalysisPassModel &>(Self).Pass;
    return typename ResultModelT::unique_ptr(
        new ResultModelT(Pass, IR, AM, std::forward<ExtraArgTs>(ExtraArgs)...));
  }

  explicit AnalysisPassModel(PassT &&Pass)
      : AnalysisPassConceptT(PassT::name(), destroyImpl, runImpl),
        Pass(std::move(Pass)) {}

public:
  static typename AnalysisPassConceptT::unique_ptr create(PassT &&Pass) {
    return typename AnalysisPassConceptT::unique_ptr(
        new AnalysisPassModel(std::move(Pass)));
  }
};

} // end namespace detail

} // end namespace llvm

#endif // LLVM_IR_PASSMANAGERINTERNAL_H

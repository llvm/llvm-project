//===- ClauseT -- clause template definitions -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef FORTRAN_LOWER_OPENMP_CLAUSET_H
#define FORTRAN_LOWER_OPENMP_CLAUSET_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/Frontend/OpenMP/OMP.h.inc"

#define ENUM(Name, ...) enum class Name { __VA_ARGS__ }
#define OPT(x) std::optional<x>

// Helper macros for enum-class conversion.
#define M(Ov, Tv)                                                              \
  if (v == OtherEnum::Ov) {                                                    \
    return ThisEnum::Tv;                                                       \
  }

#define ENUM_CONVERT(func, OtherE, ThisE, Maps)                                \
  auto func = [](OtherE v) -> ThisE {                                          \
    using ThisEnum = ThisE;                                                    \
    using OtherEnum = OtherE;                                                  \
    Maps;                                                                      \
    llvm_unreachable("Unexpected value in " #OtherE);                          \
  }

// Usage:
//
// Given two enums,
//   enum class Other { o1, o2 };
//   enum class This { t1, t2 };
// generate conversion function "Func : Other -> This" with
//   ENUM_CONVERT(Func, Other, This, M(o1, t1) M(o2, t2) ...)
//
// Note that the sequence of M(other-value, this-value) is separated
// with _spaces_, not commas.

static inline llvm::ArrayRef<llvm::omp::Directive> getWorksharing() {
  static llvm::omp::Directive worksharing[] = {
      llvm::omp::Directive::OMPD_do,     llvm::omp::Directive::OMPD_for,
      llvm::omp::Directive::OMPD_scope,  llvm::omp::Directive::OMPD_sections,
      llvm::omp::Directive::OMPD_single, llvm::omp::Directive::OMPD_workshare,
  };
  return worksharing;
}

static inline llvm::ArrayRef<llvm::omp::Directive> getWorksharingLoop() {
  static llvm::omp::Directive worksharingLoop[] = {
      llvm::omp::Directive::OMPD_do,
      llvm::omp::Directive::OMPD_for,
  };
  return worksharingLoop;
}

namespace detail {
template <typename Container, typename Predicate>
typename std::remove_reference_t<Container>::iterator
find_unique(Container &&container, Predicate &&pred) {
  auto first = std::find_if(container.begin(), container.end(), pred);
  if (first == container.end())
    return first;
  auto second = std::find_if(std::next(first), container.end(), pred);
  if (second == container.end())
    return first;
  return container.end();
}

// is_variant<T>
template <typename T>
struct is_variant {
  static constexpr bool value = false;
};

template <typename... Ts>
struct is_variant<std::variant<Ts...>> {
  static constexpr bool value = true;
};

template <typename T>
constexpr bool is_variant_v = is_variant<T>::value;

// Unions of variants
template <typename...>
struct UnionOfTwo;

template <typename... Types1, typename... Types2>
struct UnionOfTwo<std::variant<Types1...>, std::variant<Types2...>> {
  using type = std::variant<Types1..., Types2...>;
};

template <typename...>
struct Union;

template <>
struct Union<> {
  // Legal to define, illegal to instantiate.
  using type = std::variant<>;
};

template <typename T, typename... Ts>
struct Union<T, Ts...> {
  static_assert(is_variant_v<T>);
  using type = typename UnionOfTwo<T, typename Union<Ts...>::type>::type;
};
} // namespace detail

namespace tomp {

namespace type {
template <typename T>
using ListT = std::vector<T>;

// A specialization of ObjectT<Id, Expr> must provide the following definitions:
// {
//    using IdType = Id;
//    using ExprType = Expr;
//
//    auto id() const -> Id {
//      return the identifier of the object (for use in tests for
//         presence/absence of the object)
//    }
//
//    auto ref() const -> const Expr& {
//      return the expression accessing (referencing) the object
//    }
// }
//
// For example, the ObjectT instance created for "var[x+1]" would have
// the `id()` return the identifier for `var`, and the `ref()` return the
// representation of the array-access `var[x+1]`.
template <typename IdType, typename ExprType>
struct ObjectT;

template <typename I, typename E>
using ObjectListT = ListT<ObjectT<I, E>>;

using DirectiveName = llvm::omp::Directive;

template <typename I, typename E>
struct DefinedOperatorT {
  struct DefinedOpName {
    using WrapperTrait = std::true_type;
    ObjectT<I, E> v;
  };
  ENUM(IntrinsicOperator, Power, Multiply, Divide, Add, Subtract, Concat, LT,
       LE, EQ, NE, GE, GT, NOT, AND, OR, EQV, NEQV);
  using UnionTrait = std::true_type;
  std::variant<DefinedOpName, IntrinsicOperator> u;
};

template <typename E>
struct RangeT {
  // range-specification: begin : end[: step]
  using TupleTrait = std::true_type;
  std::tuple<E, E, OPT(E)> t;
};

template <typename TypeType, typename IdType, typename ExprType>
struct IteratorT {
  // iterators-specifier: [ iterator-type ] identifier = range-specification
  using TupleTrait = std::true_type;
  std::tuple<OPT(TypeType), ObjectT<IdType, ExprType>, RangeT<ExprType>> t;
};

template <typename I, typename E>
struct MapperT {
  using MapperIdentifier = ObjectT<I, E>;
  using WrapperTrait = std::true_type;
  MapperIdentifier v;
};

ENUM(MemoryOrder, AcqRel, Acquire, Relaxed, Release, SeqCst);
ENUM(MotionExpectation, Present);

template <typename I, typename E>
struct LoopIterationT {
  struct Length {
    using TupleTrait = std::true_type;
    std::tuple<DefinedOperatorT<I, E>, E> t;
  };
  using TupleTrait = std::true_type;
  std::tuple<ObjectT<I, E>, OPT(Length)> t;
};

template <typename I, typename E>
struct ProcedureDesignatorT {
  using WrapperTrait = std::true_type;
  ObjectT<I, E> v;
};

template <typename I, typename E>
struct ReductionIdentifierT {
  using UnionTrait = std::true_type;
  std::variant<DefinedOperatorT<I, E>, ProcedureDesignatorT<I, E>> u;
};
} // namespace type

template <typename T>
using ListT = type::ListT<T>;

template <typename I, typename E>
using ObjectT = type::ObjectT<I, E>;

template <typename I, typename E>
using ObjectListT = type::ObjectListT<I, E>;

namespace clause {
template <typename T, typename I, typename E>
struct AbsentT {
  using List = ListT<type::DirectiveName>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct AcqRelT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct AcquireT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct AdjustArgsT {
  using IncompleteTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct AffinityT {
  using Iterator = type::IteratorT<T, I, E>;
  using LocatorList = ObjectListT<I, E>;

  using TupleTrait = std::true_type;
  std::tuple<OPT(Iterator), LocatorList> t;
};

template <typename T, typename I, typename E>
struct AlignT {
  using Alignment = E;

  using WrapperTrait = std::true_type;
  Alignment v;
};

template <typename T, typename I, typename E>
struct AlignedT {
  using Alignment = E;
  using List = ObjectListT<I, E>;

  using TupleTrait = std::true_type;
  std::tuple<OPT(Alignment), List> t;
};

template <typename T, typename I, typename E>
struct AllocatorT;

template <typename T, typename I, typename E>
struct AllocateT {
  using AllocatorSimpleModifier = E;
  using AllocatorComplexModifier = AllocatorT<T, I, E>;
  using AlignModifier = AlignT<T, I, E>;
  using List = ObjectListT<I, E>;

  using TupleTrait = std::true_type;
  std::tuple<OPT(AllocatorSimpleModifier), OPT(AllocatorComplexModifier),
             OPT(AlignModifier), List>
      t;
};

template <typename T, typename I, typename E>
struct AllocatorT {
  using Allocator = E;
  using WrapperTrait = std::true_type;
  Allocator v;
};

template <typename T, typename I, typename E>
struct AppendArgsT {
  using IncompleteTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct AtT {
  ENUM(ActionTime, Compilation, Execution);
  using WrapperTrait = std::true_type;
  ActionTime v;
};

template <typename T, typename I, typename E>
struct AtomicDefaultMemOrderT {
  using MemoryOrder = type::MemoryOrder;
  using WrapperTrait = std::true_type;
  MemoryOrder v; // Name not provided in spec
};

template <typename T, typename I, typename E>
struct BindT {
  ENUM(Binding, Teams, Parallel, Thread);
  using WrapperTrait = std::true_type;
  Binding v;
};

template <typename T, typename I, typename E>
struct CancellationConstructTypeT {
  // Artificial
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct CaptureT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct CollapseT {
  using N = E;
  using WrapperTrait = std::true_type;
  N v;
};

template <typename T, typename I, typename E>
struct CompareT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct ContainsT {
  using List = ListT<type::DirectiveName>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct CopyinT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct CopyprivateT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct DefaultT {
  ENUM(DataSharingAttribute, Firstprivate, None, Private, Shared);
  using WrapperTrait = std::true_type;
  DataSharingAttribute v;
};

template <typename T, typename I, typename E>
struct DefaultmapT {
  ENUM(ImplicitBehavior, Alloc, To, From, Tofrom, Firstprivate, None, Default,
       Present);
  ENUM(VariableCategory, Scalar, Aggregate, Pointer, Allocatable);
  using TupleTrait = std::true_type;
  std::tuple<ImplicitBehavior, OPT(VariableCategory)> t;
};

template <typename T, typename I, typename E>
struct DoacrossT;

template <typename T, typename I, typename E>
struct DependT {
  ENUM(TaskDependenceType, In, Out, Inout, Mutexinoutset, Inoutset, Depobj);
  using Iterator = type::IteratorT<T, I, E>;
  using LocatorList = ObjectListT<I, E>;

  struct WithLocators { // Modern form
    using TupleTrait = std::true_type;
    std::tuple<TaskDependenceType, OPT(Iterator), LocatorList> t;
  };

  using Doacross = DoacrossT<T, I, E>;
  using UnionTrait = std::true_type;
  std::variant<Doacross, WithLocators> u; // Doacross form is legacy
};

template <typename T, typename I, typename E>
struct DepobjT {
  // Artificial
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct DestroyT {
  using DestroyVar = ObjectT<I, E>;
  using WrapperTrait = std::true_type;
  DestroyVar v;
};

template <typename T, typename I, typename E>
struct DetachT {
  using EventHandle = ObjectT<I, E>;
  using WrapperTrait = std::true_type;
  EventHandle v;
};

template <typename T, typename I, typename E>
struct DeviceT {
  using DeviceDescription = E;
  ENUM(DeviceModifier, Ancestor, DeviceNum);
  using TupleTrait = std::true_type;
  std::tuple<OPT(DeviceModifier), DeviceDescription> t;
};

template <typename T, typename I, typename E>
struct DeviceTypeT {
  ENUM(DeviceTypeDescription, Any, Host, Nohost);
  using WrapperTrait = std::true_type;
  DeviceTypeDescription v;
};

template <typename T, typename I, typename E>
struct DistScheduleT {
  ENUM(Kind, Static);
  using ChunkSize = E;
  using TupleTrait = std::true_type;
  std::tuple<Kind, OPT(ChunkSize)> t;
};

template <typename T, typename I, typename E>
struct DoacrossT {
  using Vector = ListT<type::LoopIterationT<I, E>>;
  ENUM(DependenceType, Source, Sink);
  using TupleTrait = std::true_type;
  std::tuple<DependenceType, Vector> t;
};

template <typename T, typename I, typename E>
struct DynamicAllocatorsT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct EnterT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct ExclusiveT {
  using WrapperTrait = std::true_type;
  using List = ObjectListT<I, E>;
  List v;
};

template <typename T, typename I, typename E>
struct FailT {
  using MemoryOrder = type::MemoryOrder;
  using WrapperTrait = std::true_type;
  MemoryOrder v;
};

template <typename T, typename I, typename E>
struct FilterT {
  using ThreadNum = E;
  using WrapperTrait = std::true_type;
  ThreadNum v;
};

template <typename T, typename I, typename E>
struct FinalT {
  using Finalize = E;
  using WrapperTrait = std::true_type;
  Finalize v;
};

template <typename T, typename I, typename E>
struct FirstprivateT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct FlushT {
  // Artificial
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct FromT {
  using LocatorList = ObjectListT<I, E>;
  using Expectation = type::MotionExpectation;
  using Mapper = type::MapperT<I, E>;
  using Iterator = type::IteratorT<T, I, E>;

  using TupleTrait = std::true_type;
  std::tuple<OPT(Expectation), OPT(Mapper), OPT(Iterator), LocatorList> t;
};

template <typename T, typename I, typename E>
struct FullT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct GrainsizeT {
  using GrainSize = E;
  using WrapperTrait = std::true_type;
  GrainSize v;
};

template <typename T, typename I, typename E>
struct HasDeviceAddrT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct HintT {
  using HintExpr = E;
  using WrapperTrait = std::true_type;
  HintExpr v;
};

template <typename T, typename I, typename E>
struct HoldsT {
  using WrapperTrait = std::true_type;
  E v; // No argument name in spec 5.2
};

template <typename T, typename I, typename E>
struct IfT {
  using DirectiveNameModifier = type::DirectiveName;
  using IfExpression = E;
  using TupleTrait = std::true_type;
  std::tuple<OPT(DirectiveNameModifier), IfExpression> t;
};

template <typename T, typename I, typename E>
struct InbranchT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct InclusiveT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct IndirectT {
  using InvokedByFptr = E;
  using WrapperTrait = std::true_type;
  InvokedByFptr v;
};

template <typename T, typename I, typename E>
struct InitT {
  using ForeignRuntimeId = E;
  using InteropVar = ObjectT<I, E>;
  using InteropPreference = ListT<ForeignRuntimeId>;
  ENUM(InteropType, Target, Targetsync);

  using TupleTrait = std::true_type;
  std::tuple<OPT(InteropPreference), InteropType, InteropVar> t;
};

template <typename T, typename I, typename E>
struct InitializerT {
  using InitializerExpr = E;
  using WrapperTrait = std::true_type;
  InitializerExpr v;
};

template <typename T, typename I, typename E>
struct InReductionT {
  using List = ObjectListT<I, E>;
  using ReductionIdentifier = type::ReductionIdentifierT<I, E>;
  using TupleTrait = std::true_type;
  std::tuple<ReductionIdentifier, List> t;
};

template <typename T, typename I, typename E>
struct IsDevicePtrT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct LastprivateT {
  using List = ObjectListT<I, E>;
  ENUM(LastprivateModifier, Conditional);
  using TupleTrait = std::true_type;
  std::tuple<OPT(LastprivateModifier), List> t;
};

template <typename T, typename I, typename E>
struct LinearT {
  // std::get<type> won't work here due to duplicate types in the tuple.
  using List = ObjectListT<I, E>;
  using StepSimpleModifier = E;
  using StepComplexModifier = E;
  ENUM(LinearModifier, Ref, Val, Uval);

  using TupleTrait = std::true_type;
  std::tuple<OPT(StepSimpleModifier), OPT(StepComplexModifier),
             OPT(LinearModifier), List>
      t;
};

template <typename T, typename I, typename E>
struct LinkT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct MapT {
  using LocatorList = ObjectListT<I, E>;
  ENUM(MapType, To, From, Tofrom, Alloc, Release, Delete);
  ENUM(MapTypeModifier, Always, Close, Present);
  using Mapper = type::MapperT<I, E>;
  using Iterator = type::IteratorT<T, I, E>;

  using TupleTrait = std::true_type;
  std::tuple<OPT(MapType), OPT(MapTypeModifier), OPT(Mapper), OPT(Iterator),
             LocatorList>
      t;
};

template <typename T, typename I, typename E>
struct MatchT {
  using IncompleteTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct MemoryOrderT {
  // Artificial
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct MergeableT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct MessageT {
  using MsgString = E;
  using WrapperTrait = std::true_type;
  MsgString v;
};

template <typename T, typename I, typename E>
struct NocontextT {
  using DoNotUpdateContext = E;
  using WrapperTrait = std::true_type;
  DoNotUpdateContext v;
};

template <typename T, typename I, typename E>
struct NogroupT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct NontemporalT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct NoOpenmp {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct NoOpenmpRoutines {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct NoParallelism {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct NotinbranchT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct NovariantsT {
  using DoNotUseVariant = E;
  using WrapperTrait = std::true_type;
  DoNotUseVariant v;
};

template <typename T, typename I, typename E>
struct NowaitT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct NumTasksT {
  using NumTasks = E;
  ENUM(Prescriptiveness, Strict);
  using TupleTrait = std::true_type;
  std::tuple<OPT(Prescriptiveness), NumTasks> t;
};

template <typename T, typename I, typename E>
struct NumTeamsT {
  using TupleTrait = std::true_type;
  using LowerBound = E;
  using UpperBound = E;
  std::tuple<OPT(LowerBound), UpperBound> t;
};

template <typename T, typename I, typename E>
struct NumThreadsT {
  using Nthreads = E;
  using WrapperTrait = std::true_type;
  Nthreads v;
};

template <typename T, typename I, typename E>
struct OmpxAttributeT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct OmpxBareT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct OmpxDynCgroupMemT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename T, typename I, typename E>
struct OrderT {
  ENUM(OrderModifier, Reproducible, Unconstrained);
  ENUM(Ordering, Concurrent);
  using TupleTrait = std::true_type;
  std::tuple<OPT(OrderModifier), Ordering> t;
};

template <typename T, typename I, typename E>
struct OrderedT {
  using N = E;
  using WrapperTrait = std::true_type;
  OPT(N) v;
};

template <typename T, typename I, typename E>
struct OtherwiseT {
  using IncompleteTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct PartialT {
  using UnrollFactor = E;
  using WrapperTrait = std::true_type;
  OPT(UnrollFactor) v;
};

template <typename T, typename I, typename E>
struct PriorityT {
  using PriorityValue = E;
  using WrapperTrait = std::true_type;
  PriorityValue v;
};

template <typename T, typename I, typename E>
struct PrivateT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct ProcBindT {
  ENUM(AffinityPolicy, Close, Master, Spread, Primary);
  using WrapperTrait = std::true_type;
  AffinityPolicy v;
};

template <typename T, typename I, typename E>
struct ReadT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct ReductionT {
  using List = ObjectListT<I, E>;
  using ReductionIdentifier = type::ReductionIdentifierT<I, E>;
  ENUM(ReductionModifier, Default, Inscan, Task);
  using TupleTrait = std::true_type;
  std::tuple<ReductionIdentifier, OPT(ReductionModifier), List> t;
};

template <typename T, typename I, typename E>
struct RelaxedT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct ReleaseT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct ReverseOffloadT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct SafelenT {
  using Length = E;
  using WrapperTrait = std::true_type;
  Length v;
};

template <typename T, typename I, typename E>
struct ScheduleT {
  ENUM(Kind, Static, Dynamic, Guided, Auto, Runtime);
  using ChunkSize = E;
  ENUM(OrderingModifier, Monotonic, Nonmonotonic);
  ENUM(ChunkModifier, Simd);
  using TupleTrait = std::true_type;
  std::tuple<Kind, OPT(OrderingModifier), OPT(ChunkModifier), OPT(ChunkSize)> t;
};

template <typename T, typename I, typename E>
struct SeqCstT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct SeverityT {
  ENUM(SevLevel, Fatal, Warning);
  using WrapperTrait = std::true_type;
  SevLevel v;
};

template <typename T, typename I, typename E>
struct SharedT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct SimdT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct SimdlenT {
  using Length = E;
  using WrapperTrait = std::true_type;
  Length v;
};

template <typename T, typename I, typename E>
struct SizesT {
  using SizeList = ListT<E>;
  using WrapperTrait = std::true_type;
  SizeList v;
};

template <typename T, typename I, typename E>
struct TaskReductionT {
  using List = ObjectListT<I, E>;
  using ReductionIdentifier = type::ReductionIdentifierT<I, E>;
  using TupleTrait = std::true_type;
  std::tuple<ReductionIdentifier, List> t;
};

template <typename T, typename I, typename E>
struct ThreadLimitT {
  using Threadlim = E;
  using WrapperTrait = std::true_type;
  Threadlim v;
};

template <typename T, typename I, typename E>
struct ThreadprivateT {
  // Artificial
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct ThreadsT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct ToT {
  using LocatorList = ObjectListT<I, E>;
  using Expectation = type::MotionExpectation;
  using Mapper = type::MapperT<I, E>;
  using Iterator = type::IteratorT<T, I, E>;

  using TupleTrait = std::true_type;
  std::tuple<OPT(Expectation), OPT(Mapper), OPT(Iterator), LocatorList> t;
};

template <typename T, typename I, typename E>
struct UnifiedAddressT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct UnifiedSharedMemoryT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct UniformT {
  using ParameterList = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  ParameterList v;
};

template <typename T, typename I, typename E>
struct UnknownT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct UntiedT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct UpdateT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct UseT {
  using InteropVar = ObjectT<I, E>;
  using WrapperTrait = std::true_type;
  InteropVar v;
};

template <typename T, typename I, typename E>
struct UseDeviceAddrT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct UseDevicePtrT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E>
struct UsesAllocatorsT {
  using MemSpace = E;
  using TraitsArray = ObjectT<I, E>;
  using Allocator = E;
  using TupleTrait = std::true_type;
  std::tuple<OPT(MemSpace), OPT(TraitsArray), Allocator> t;
};

template <typename T, typename I, typename E>
struct WeakT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct WhenT {
  using IncompleteTrait = std::true_type;
};

template <typename T, typename I, typename E>
struct WriteT {
  using EmptyTrait = std::true_type;
};

// ---

template <typename T, typename I, typename E>
using ArtificialClausesT =
    std::variant<CancellationConstructTypeT<T, I, E>, DepobjT<T, I, E>,
                 FlushT<T, I, E>, MemoryOrderT<T, I, E>,
                 ThreadprivateT<T, I, E>>;

template <typename T, typename I, typename E>
using EmptyClausesT = std::variant<
    AcqRelT<T, I, E>, AcquireT<T, I, E>, CaptureT<T, I, E>, CompareT<T, I, E>,
    DynamicAllocatorsT<T, I, E>, FullT<T, I, E>, InbranchT<T, I, E>,
    MergeableT<T, I, E>, NogroupT<T, I, E>, NoOpenmpRoutines<T, I, E>,
    NoOpenmp<T, I, E>, NoParallelism<T, I, E>, NotinbranchT<T, I, E>,
    NowaitT<T, I, E>, OmpxAttributeT<T, I, E>, OmpxBareT<T, I, E>,
    ReadT<T, I, E>, RelaxedT<T, I, E>, ReleaseT<T, I, E>,
    ReverseOffloadT<T, I, E>, SeqCstT<T, I, E>, SimdT<T, I, E>,
    ThreadsT<T, I, E>, UnifiedAddressT<T, I, E>, UnifiedSharedMemoryT<T, I, E>,
    UnknownT<T, I, E>, UntiedT<T, I, E>, UpdateT<T, I, E>, UseT<T, I, E>,
    WeakT<T, I, E>, WriteT<T, I, E>>;

template <typename T, typename I, typename E>
using IncompleteClausesT =
    std::variant<AdjustArgsT<T, I, E>, AppendArgsT<T, I, E>, MatchT<T, I, E>,
                 OtherwiseT<T, I, E>, WhenT<T, I, E>>;

template <typename T, typename I, typename E>
using TupleClausesT =
    std::variant<AffinityT<T, I, E>, AlignedT<T, I, E>, AllocateT<T, I, E>,
                 DefaultmapT<T, I, E>, DeviceT<T, I, E>, DistScheduleT<T, I, E>,
                 DoacrossT<T, I, E>, FromT<T, I, E>, IfT<T, I, E>,
                 InitT<T, I, E>, InReductionT<T, I, E>, LastprivateT<T, I, E>,
                 LinearT<T, I, E>, MapT<T, I, E>, NumTasksT<T, I, E>,
                 OrderT<T, I, E>, ReductionT<T, I, E>, ScheduleT<T, I, E>,
                 TaskReductionT<T, I, E>, ToT<T, I, E>,
                 UsesAllocatorsT<T, I, E>>;

template <typename T, typename I, typename E>
using UnionClausesT = std::variant<DependT<T, I, E>>;

template <typename T, typename I, typename E>
using WrapperClausesT = std::variant<
    AbsentT<T, I, E>, AlignT<T, I, E>, AllocatorT<T, I, E>,
    AtomicDefaultMemOrderT<T, I, E>, AtT<T, I, E>, BindT<T, I, E>,
    CollapseT<T, I, E>, ContainsT<T, I, E>, CopyinT<T, I, E>,
    CopyprivateT<T, I, E>, DefaultT<T, I, E>, DestroyT<T, I, E>,
    DetachT<T, I, E>, DeviceTypeT<T, I, E>, EnterT<T, I, E>,
    ExclusiveT<T, I, E>, FailT<T, I, E>, FilterT<T, I, E>, FinalT<T, I, E>,
    FirstprivateT<T, I, E>, GrainsizeT<T, I, E>, HasDeviceAddrT<T, I, E>,
    HintT<T, I, E>, HoldsT<T, I, E>, InclusiveT<T, I, E>, IndirectT<T, I, E>,
    InitializerT<T, I, E>, IsDevicePtrT<T, I, E>, LinkT<T, I, E>,
    MessageT<T, I, E>, NocontextT<T, I, E>, NontemporalT<T, I, E>,
    NovariantsT<T, I, E>, NumTeamsT<T, I, E>, NumThreadsT<T, I, E>,
    OmpxDynCgroupMemT<T, I, E>, OrderedT<T, I, E>, PartialT<T, I, E>,
    PriorityT<T, I, E>, PrivateT<T, I, E>, ProcBindT<T, I, E>,
    SafelenT<T, I, E>, SeverityT<T, I, E>, SharedT<T, I, E>, SimdlenT<T, I, E>,
    SizesT<T, I, E>, ThreadLimitT<T, I, E>, UniformT<T, I, E>,
    UseDeviceAddrT<T, I, E>, UseDevicePtrT<T, I, E>>;

template <typename T, typename I, typename E>
using UnionOfAllClausesT = typename detail::Union< //
    ArtificialClausesT<T, I, E>,                   //
    EmptyClausesT<T, I, E>,                        //
    IncompleteClausesT<T, I, E>,                   //
    TupleClausesT<T, I, E>,                        //
    UnionClausesT<T, I, E>,                        //
    WrapperClausesT<T, I, E>                       //
    >::type;

} // namespace clause

template <typename TypeType, typename IdType, typename ExprType>
struct ClauseT {
  llvm::omp::Clause id; // The numeric id of the clause
  using UnionTrait = std::true_type;
  clause::UnionOfAllClausesT<TypeType, IdType, ExprType> u;
};

// --------------------------------------------------------------------
template <typename TypeType, typename IdType, typename ExprType>
struct DirectiveInfo {
  llvm::omp::Directive id = llvm::omp::Directive::OMPD_unknown;
  llvm::SmallVector<const tomp::ClauseT<TypeType, IdType, ExprType> *> clauses;
};

template <typename T, typename I, typename E>
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const DirectiveInfo<T, I, E> &di) {
  os << llvm::omp::getOpenMPDirectiveName(di.id) << " -> {";
  for (int i = 0, e = di.clauses.size(); i != e; ++i) {
    if (i != 0)
      os << ", ";
    os << llvm::omp::getOpenMPClauseName(di.clauses[i]->id);
  }
  os << '}';
  return os;
}

template <typename TypeType, typename IdType, typename ExprType,
          typename HelperType>
struct CompositeInfoBase {
  using IdTy = IdType;
  using ExprTy = ExprType;
  using TypeTy = TypeType;
  using HelperTy = HelperType;
  using ObjectTy = tomp::ObjectT<IdTy, ExprTy>;
  using ClauseTy = tomp::ClauseT<TypeTy, IdTy, ExprTy>;

  using ClauseSet = llvm::DenseSet<const ClauseTy *>;

  CompositeInfoBase(uint32_t ver, llvm::omp::Directive dir, HelperType &hlp)
      : version(ver), construct(dir), helper(hlp) {}

  void add(const ClauseTy *node) { nodes.push_back(node); }

  bool split();

  DirectiveInfo<TypeTy, IdTy, ExprTy> *
  findDirective(llvm::omp::Directive dirId) {
    for (DirectiveInfo<TypeTy, IdTy, ExprTy> &dir : leafs) {
      if (dir.id == dirId)
        return &dir;
    }
    return nullptr;
  }
  ClauseSet *findClauses(const ObjectTy &object) {
    if (auto found = syms.find(object.id()); found != syms.end())
      return &found->second;
    return nullptr;
  }

  uint32_t version;
  llvm::omp::Directive construct;

  // Leafs are ordered outer to inner.
  llvm::SmallVector<DirectiveInfo<TypeTy, IdTy, ExprTy>> leafs;

private:
  template <typename S>
  ClauseTy *makeClause(llvm::omp::Clause clauseId, S &&specific) {
    auto clause = ClauseTy{clauseId, std::move(specific)};
    auto *addr = helper.makeClause(std::move(clause));
    return static_cast<ClauseTy *>(addr);
  }

  void addClauseSymsToMap(const ObjectTy &object, const ClauseTy *);
  void addClauseSymsToMap(const tomp::ObjectListT<IdTy, ExprTy> &objects,
                          const ClauseTy *);
  void addClauseSymsToMap(const TypeTy &item, const ClauseTy *);
  void addClauseSymsToMap(const ExprTy &item, const ClauseTy *);
  void addClauseSymsToMap(const tomp::clause::MapT<TypeTy, IdTy, ExprTy> &item,
                          const ClauseTy *);

  template <typename U>
  void addClauseSymsToMap(const std::optional<U> &item, const ClauseTy *);
  template <typename U>
  void addClauseSymsToMap(const tomp::ListT<U> &item, const ClauseTy *);
  template <typename... U, size_t... Is>
  void addClauseSymsToMap(const std::tuple<U...> &item, const ClauseTy *,
                          std::index_sequence<Is...> = {});
  template <typename U>
  std::enable_if_t<std::is_enum_v<llvm::remove_cvref_t<U>>, void>
  addClauseSymsToMap(U &&item, const ClauseTy *);

  template <typename U>
  std::enable_if_t<llvm::remove_cvref_t<U>::EmptyTrait::value, void>
  addClauseSymsToMap(U &&item, const ClauseTy *);

  template <typename U>
  std::enable_if_t<llvm::remove_cvref_t<U>::IncompleteTrait::value, void>
  addClauseSymsToMap(U &&item, const ClauseTy *);

  template <typename U>
  std::enable_if_t<llvm::remove_cvref_t<U>::WrapperTrait::value, void>
  addClauseSymsToMap(U &&item, const ClauseTy *);

  template <typename U>
  std::enable_if_t<llvm::remove_cvref_t<U>::TupleTrait::value, void>
  addClauseSymsToMap(U &&item, const ClauseTy *);

  template <typename U>
  std::enable_if_t<llvm::remove_cvref_t<U>::UnionTrait::value, void>
  addClauseSymsToMap(U &&item, const ClauseTy *);

  // Apply a clause to the only directive that allows it. If there are no
  // directives that allow it, or if there is more that one, do not apply
  // anything and return false, otherwise return true.
  bool applyToUnique(const ClauseTy *node);

  // Apply a clause to the first directive in given range that allows it.
  // If such a directive does not exist, return false, otherwise return true.
  template <typename Iterator>
  bool applyToFirst(const ClauseTy *node, llvm::iterator_range<Iterator> range);

  // Apply a clause to the innermost directive that allows it. If such a
  // directive does not exist, return false, otherwise return true.
  bool applyToInnermost(const ClauseTy *node);

  // Apply a clause to the outermost directive that allows it. If such a
  // directive does not exist, return false, otherwise return true.
  bool applyToOutermost(const ClauseTy *node);

  template <typename Predicate>
  bool applyIf(const ClauseTy *node, Predicate shouldApply);

  bool applyToAll(const ClauseTy *node);

  template <typename Clause>
  bool applyClause(Clause &&clause, const ClauseTy *node);

  bool applyClause(const tomp::clause::CollapseT<TypeTy, IdTy, ExprTy> &clause,
                   const ClauseTy *);
  bool applyClause(const tomp::clause::PrivateT<TypeTy, IdTy, ExprTy> &clause,
                   const ClauseTy *);
  bool
  applyClause(const tomp::clause::FirstprivateT<TypeTy, IdTy, ExprTy> &clause,
              const ClauseTy *);
  bool
  applyClause(const tomp::clause::LastprivateT<TypeTy, IdTy, ExprTy> &clause,
              const ClauseTy *);
  bool applyClause(const tomp::clause::SharedT<TypeTy, IdTy, ExprTy> &clause,
                   const ClauseTy *);
  bool applyClause(const tomp::clause::DefaultT<TypeTy, IdTy, ExprTy> &clause,
                   const ClauseTy *);
  bool
  applyClause(const tomp::clause::ThreadLimitT<TypeTy, IdTy, ExprTy> &clause,
              const ClauseTy *);
  bool applyClause(const tomp::clause::OrderT<TypeTy, IdTy, ExprTy> &clause,
                   const ClauseTy *);
  bool applyClause(const tomp::clause::AllocateT<TypeTy, IdTy, ExprTy> &clause,
                   const ClauseTy *);
  bool applyClause(const tomp::clause::ReductionT<TypeTy, IdTy, ExprTy> &clause,
                   const ClauseTy *);
  bool applyClause(const tomp::clause::IfT<TypeTy, IdTy, ExprTy> &clause,
                   const ClauseTy *);
  bool applyClause(const tomp::clause::LinearT<TypeTy, IdTy, ExprTy> &clause,
                   const ClauseTy *);
  bool applyClause(const tomp::clause::NowaitT<TypeTy, IdTy, ExprTy> &clause,
                   const ClauseTy *);

  HelperType &helper;
  tomp::ListT<const ClauseTy *> nodes;

  llvm::DenseMap<IdTy, ClauseSet> syms;
  llvm::DenseSet<IdTy> mapBases;
};

template <typename T, typename I, typename E, typename H>
void CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(const ObjectTy &object,
                                                       const ClauseTy *node) {
  syms[object.id()].insert(node);
}

template <typename T, typename I, typename E, typename H>
void CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(
    const tomp::ObjectListT<I, E> &objects, const ClauseTy *node) {
  for (auto &object : objects)
    syms[object.id()].insert(node);
}

template <typename T, typename I, typename E, typename H>
void CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(const T &item,
                                                       const ClauseTy *node) {
  // Nothing to do for types.
}

template <typename T, typename I, typename E, typename H>
void CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(const E &item,
                                                       const ClauseTy *node) {
  // Nothing to do for expressions.
}

template <typename T, typename I, typename E, typename H>
void CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(
    const tomp::clause::MapT<T, I, E> &item, const ClauseTy *node) {
  auto &objects = std::get<tomp::ObjectListT<I, E>>(item.t);
  addClauseSymsToMap(objects, node);
  for (auto &object : objects) {
    if (auto base = helper.getBaseObject(object))
      mapBases.insert(base->id());
  }
}

template <typename T, typename I, typename E, typename H>
template <typename U>
void CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(
    const std::optional<U> &item, const ClauseTy *node) {
  if (item)
    addClauseSymsToMap(*item, node);
}

template <typename T, typename I, typename E, typename H>
template <typename U>
void CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(
    const tomp::ListT<U> &item, const ClauseTy *node) {
  for (auto &s : item)
    addClauseSymsToMap(s, node);
}

template <typename T, typename I, typename E, typename H>
template <typename... U, size_t... Is>
void CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(
    const std::tuple<U...> &item, const ClauseTy *node,
    std::index_sequence<Is...>) {
  (void)node; // Silence strange warning from GCC.
  (addClauseSymsToMap(std::get<Is>(item), node), ...);
}

template <typename T, typename I, typename E, typename H>
template <typename U>
std::enable_if_t<std::is_enum_v<llvm::remove_cvref_t<U>>, void>
CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(U &&item,
                                                  const ClauseTy *node) {
  // Nothing to do for enums.
}

template <typename T, typename I, typename E, typename H>
template <typename U>
std::enable_if_t<llvm::remove_cvref_t<U>::EmptyTrait::value, void>
CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(U &&item,
                                                  const ClauseTy *node) {
  // Nothing to do for an empty class.
}

template <typename T, typename I, typename E, typename H>
template <typename U>
std::enable_if_t<llvm::remove_cvref_t<U>::IncompleteTrait::value, void>
CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(U &&item,
                                                  const ClauseTy *node) {
  // Nothing to do for an incomplete class (they're empty).
}

template <typename T, typename I, typename E, typename H>
template <typename U>
std::enable_if_t<llvm::remove_cvref_t<U>::WrapperTrait::value, void>
CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(U &&item,
                                                  const ClauseTy *node) {
  addClauseSymsToMap(item.v, node);
}

template <typename T, typename I, typename E, typename H>
template <typename U>
std::enable_if_t<llvm::remove_cvref_t<U>::TupleTrait::value, void>
CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(U &&item,
                                                  const ClauseTy *node) {
  constexpr size_t tuple_size =
      std::tuple_size_v<llvm::remove_cvref_t<decltype(item.t)>>;
  addClauseSymsToMap(item.t, node, std::make_index_sequence<tuple_size>{});
}

template <typename T, typename I, typename E, typename H>
template <typename U>
std::enable_if_t<llvm::remove_cvref_t<U>::UnionTrait::value, void>
CompositeInfoBase<T, I, E, H>::addClauseSymsToMap(U &&item,
                                                  const ClauseTy *node) {
  std::visit([&](auto &&s) { addClauseSymsToMap(s, node); }, item.u);
}

// Apply a clause to the only directive that allows it. If there are no
// directives that allow it, or if there is more that one, do not apply
// anything and return false, otherwise return true.
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyToUnique(const ClauseTy *node) {
  auto unique = detail::find_unique(leafs, [=](const auto &dirInfo) {
    return llvm::omp::isAllowedClauseForDirective(dirInfo.id, node->id,
                                                  version);
  });

  if (unique != leafs.end()) {
    unique->clauses.push_back(node);
    return true;
  }
  return false;
}

// Apply a clause to the first directive in given range that allows it.
// If such a directive does not exist, return false, otherwise return true.
template <typename T, typename I, typename E, typename H>
template <typename Iterator>
bool CompositeInfoBase<T, I, E, H>::applyToFirst(
    const ClauseTy *node, llvm::iterator_range<Iterator> range) {
  if (range.empty())
    return false;

  for (DirectiveInfo<T, I, E> &dir : range) {
    if (!llvm::omp::isAllowedClauseForDirective(dir.id, node->id, version))
      continue;
    dir.clauses.push_back(node);
    return true;
  }
  return false;
}

// Apply a clause to the innermost directive that allows it. If such a
// directive does not exist, return false, otherwise return true.
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyToInnermost(const ClauseTy *node) {
  return applyToFirst(node, llvm::reverse(leafs));
}

// Apply a clause to the outermost directive that allows it. If such a
// directive does not exist, return false, otherwise return true.
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyToOutermost(const ClauseTy *node) {
  return applyToFirst(node, llvm::iterator_range(leafs));
}

template <typename T, typename I, typename E, typename H>
template <typename Predicate>
bool CompositeInfoBase<T, I, E, H>::applyIf(const ClauseTy *node,
                                            Predicate shouldApply) {
  bool applied = false;
  for (DirectiveInfo<T, I, E> &dir : leafs) {
    if (!llvm::omp::isAllowedClauseForDirective(dir.id, node->id, version))
      continue;
    if (!shouldApply(dir))
      continue;
    dir.clauses.push_back(node);
    applied = true;
  }

  return applied;
}

template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyToAll(const ClauseTy *node) {
  return applyIf(node, [](auto) { return true; });
}

template <typename T, typename I, typename E, typename H>
template <typename Clause>
bool CompositeInfoBase<T, I, E, H>::applyClause(Clause &&clause,
                                                const ClauseTy *node) {
  // The default behavior is to find the unique directive to which the
  // given clause may be applied. If there are no such directives, or
  // if there are multiple ones, flag an error.
  // From "OpenMP Application Programming Interface", Version 5.2:
  // S Some clauses are permitted only on a single leaf construct of the
  // S combined or composite construct, in which case the effect is as if
  // S the clause is applied to that specific construct. (p339, 31-33)
  if (applyToUnique(node))
    return true;

  return false;
}

// COLLAPSE
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::CollapseT<T, I, E> &clause, const ClauseTy *node) {
  // Apply COLLAPSE to the innermost directive. If it's not one that
  // allows it flag an error.
  if (!leafs.empty()) {
    DirectiveInfo<T, I, E> &last = leafs.back();

    if (llvm::omp::isAllowedClauseForDirective(last.id, node->id, version)) {
      last.clauses.push_back(node);
      return true;
    }
  }

  // llvm::errs() << "Cannot apply COLLAPSE\n";
  return false;
}

// PRIVATE
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::PrivateT<T, I, E> &clause, const ClauseTy *node) {
  if (applyToInnermost(node))
    return true;
  // llvm::errs() << "Cannot apply PRIVATE\n";
  return false;
}

// FIRSTPRIVATE
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::FirstprivateT<T, I, E> &clause, const ClauseTy *node) {
  bool applied = false;

  // S Section 17.2
  // S The effect of the firstprivate clause is as if it is applied to one
  // S or more leaf constructs as follows:

  // S - To the distribute construct if it is among the constituent constructs;
  // S - To the teams construct if it is among the constituent constructs and
  // S   the distribute construct is not;
  auto hasDistribute = findDirective(llvm::omp::OMPD_distribute);
  auto hasTeams = findDirective(llvm::omp::OMPD_teams);
  if (hasDistribute != nullptr) {
    hasDistribute->clauses.push_back(node);
    applied = true;
    // S If the teams construct is among the constituent constructs and the
    // S effect is not as if the firstprivate clause is applied to it by the
    // S above rules, then the effect is as if the shared clause with the
    // S same list item is applied to the teams construct.
    if (hasTeams != nullptr) {
      auto *shared =
          makeClause(llvm::omp::Clause::OMPC_shared,
                     tomp::clause::SharedT<T, I, E>{/*List=*/clause.v});
      hasTeams->clauses.push_back(shared);
    }
  } else if (hasTeams != nullptr) {
    hasTeams->clauses.push_back(node);
    applied = true;
  }

  // S - To a worksharing construct that accepts the clause if one is among
  // S   the constituent constructs;
  auto findWorksharing = [&]() {
    auto worksharing = getWorksharing();
    for (DirectiveInfo<T, I, E> &dir : leafs) {
      auto found = llvm::find(worksharing, dir.id);
      if (found != std::end(worksharing))
        return &dir;
    }
    return static_cast<DirectiveInfo<T, I, E> *>(nullptr);
  };

  auto hasWorksharing = findWorksharing();
  if (hasWorksharing != nullptr) {
    hasWorksharing->clauses.push_back(node);
    applied = true;
  }

  // S - To the taskloop construct if it is among the constituent constructs;
  auto hasTaskloop = findDirective(llvm::omp::OMPD_taskloop);
  if (hasTaskloop != nullptr) {
    hasTaskloop->clauses.push_back(node);
    applied = true;
  }

  // S - To the parallel construct if it is among the constituent constructs
  // S   and neither a taskloop construct nor a worksharing construct that
  // S   accepts the clause is among them;
  auto hasParallel = findDirective(llvm::omp::OMPD_parallel);
  if (hasParallel != nullptr) {
    if (hasTaskloop == nullptr && hasWorksharing == nullptr) {
      hasParallel->clauses.push_back(node);
      applied = true;
    } else {
      // S If the parallel construct is among the constituent constructs and
      // S the effect is not as if the firstprivate clause is applied to it by
      // S the above rules, then the effect is as if the shared clause with
      // S the same list item is applied to the parallel construct.
      auto *shared =
          makeClause(llvm::omp::Clause::OMPC_shared,
                     tomp::clause::SharedT<T, I, E>{/*List=*/clause.v});
      hasParallel->clauses.push_back(shared);
    }
  }

  // S - To the target construct if it is among the constituent constructs
  // S   and the same list item neither appears in a lastprivate clause nor
  // S   is the base variable or base pointer of a list item that appears in
  // S   a map clause.
  auto inLastprivate = [&](const ObjectTy &object) {
    if (ClauseSet *set = findClauses(object)) {
      return llvm::find_if(*set, [](const ClauseTy *c) {
               return c->id == llvm::omp::Clause::OMPC_lastprivate;
             }) != set->end();
    }
    return false;
  };

  auto hasTarget = findDirective(llvm::omp::OMPD_target);
  if (hasTarget != nullptr) {
    tomp::ObjectListT<I, E> objects;
    llvm::copy_if(
        clause.v, std::back_inserter(objects), [&](const ObjectTy &object) {
          return !inLastprivate(object) && !mapBases.contains(object.id());
        });
    if (!objects.empty()) {
      auto *firstp =
          makeClause(llvm::omp::Clause::OMPC_firstprivate,
                     tomp::clause::FirstprivateT<T, I, E>{/*List=*/objects});
      hasTarget->clauses.push_back(firstp);
      applied = true;
    }
  }

  return applied;
}

// LASTPRIVATE
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::LastprivateT<T, I, E> &clause, const ClauseTy *node) {
  bool applied = false;

  // S The effect of the lastprivate clause is as if it is applied to all leaf
  // S constructs that permit the clause.
  if (!applyToAll(node)) {
    // llvm::errs() << "Cannot apply LASTPRIVATE\n";
    return false;
  }

  auto inFirstprivate = [&](const ObjectTy &object) {
    if (ClauseSet *set = findClauses(object)) {
      return llvm::find_if(*set, [](const ClauseTy *c) {
               return c->id == llvm::omp::Clause::OMPC_firstprivate;
             }) != set->end();
    }
    return false;
  };

  auto &objects = std::get<tomp::ObjectListT<I, E>>(clause.t);

  // Prepare list of objects that could end up in a SHARED clause.
  tomp::ObjectListT<I, E> sharedObjects;
  llvm::copy_if(
      objects, std::back_inserter(sharedObjects),
      [&](const ObjectTy &object) { return !inFirstprivate(object); });

  if (!sharedObjects.empty()) {
    // S If the parallel construct is among the constituent constructs and the
    // S list item is not also specified in the firstprivate clause, then the
    // S effect of the lastprivate clause is as if the shared clause with the
    // S same list item is applied to the parallel construct.
    if (auto hasParallel = findDirective(llvm::omp::OMPD_parallel)) {
      auto *shared =
          makeClause(llvm::omp::Clause::OMPC_shared,
                     tomp::clause::SharedT<T, I, E>{/*List=*/sharedObjects});
      hasParallel->clauses.push_back(shared);
      applied = true;
    }

    // S If the teams construct is among the constituent constructs and the
    // S list item is not also specified in the firstprivate clause, then the
    // S effect of the lastprivate clause is as if the shared clause with the
    // S same list item is applied to the teams construct.
    if (auto hasTeams = findDirective(llvm::omp::OMPD_teams)) {
      auto *shared =
          makeClause(llvm::omp::Clause::OMPC_shared,
                     tomp::clause::SharedT<T, I, E>{/*List=*/sharedObjects});
      hasTeams->clauses.push_back(shared);
      applied = true;
    }
  }

  // S If the target construct is among the constituent constructs and the
  // S list item is not the base variable or base pointer of a list item that
  // S appears in a map clause, the effect of the lastprivate clause is as if
  // S the same list item appears in a map clause with a map-type of tofrom.
  if (auto hasTarget = findDirective(llvm::omp::OMPD_target)) {
    tomp::ObjectListT<I, E> tofrom;
    llvm::copy_if(objects, std::back_inserter(tofrom),
                  [&](const ObjectTy &object) {
                    return !mapBases.contains(object.id());
                  });

    if (!tofrom.empty()) {
      using MapType = typename tomp::clause::MapT<T, I, E>::MapType;
      auto *map =
          makeClause(llvm::omp::Clause::OMPC_map,
                     tomp::clause::MapT<T, I, E>{
                         {/*MapType=*/MapType::Tofrom,
                          /*MapTypeModifier=*/std::nullopt,
                          /*Mapper=*/std::nullopt, /*Iterator=*/std::nullopt,
                          /*LocatorList=*/std::move(tofrom)}});
      hasTarget->clauses.push_back(map);
      applied = true;
    }
  }

  return applied;
}

// SHARED
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::SharedT<T, I, E> &clause, const ClauseTy *node) {
  // Apply SHARED to the all leafs that allow it.
  if (applyToAll(node))
    return true;
  // llvm::errs() << "Cannot apply SHARED\n";
  return false;
}

// DEFAULT
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::DefaultT<T, I, E> &clause, const ClauseTy *node) {
  // Apply DEFAULT to the all leafs that allow it.
  if (applyToAll(node))
    return true;
  // llvm::errs() << "Cannot apply DEFAULT\n";
  return false;
}

// THREAD_LIMIT
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::ThreadLimitT<T, I, E> &clause, const ClauseTy *node) {
  // Apply THREAD_LIMIT to the all leafs that allow it.
  if (applyToAll(node))
    return true;
  // llvm::errs() << "Cannot apply THREAD_LIMIT\n";
  return false;
}

// ORDER
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::OrderT<T, I, E> &clause, const ClauseTy *node) {
  // Apply ORDER to the all leafs that allow it.
  if (applyToAll(node))
    return true;
  // llvm::errs() << "Cannot apply ORDER\n";
  return false;
}

// ALLOCATE
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::AllocateT<T, I, E> &clause, const ClauseTy *node) {
  // This one needs to be applied at the end, once we know which clauses are
  // assigned to which leaf constructs.

  // S The effect of the allocate clause is as if it is applied to all leaf
  // S constructs that permit the clause and to which a data-sharing attribute
  // S clause that may create a private copy of the same list item is applied.

  auto canMakePrivateCopy = [](llvm::omp::Clause id) {
    switch (id) {
    case llvm::omp::Clause::OMPC_firstprivate:
    case llvm::omp::Clause::OMPC_lastprivate:
    case llvm::omp::Clause::OMPC_private:
      return true;
    default:
      return false;
    }
  };

  bool applied = applyIf(node, [&](const DirectiveInfo<T, I, E> &dir) {
    return llvm::any_of(dir.clauses, [&](const ClauseTy *n) {
      return canMakePrivateCopy(n->id);
    });
  });

  return applied;
}

// REDUCTION
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::ReductionT<T, I, E> &clause, const ClauseTy *node) {
  // S The effect of the reduction clause is as if it is applied to all leaf
  // S constructs that permit the clause, except for the following constructs:
  // S - The parallel construct, when combined with the sections, worksharing-
  // S   loop, loop, or taskloop construct; and
  // S - The teams construct, when combined with the loop construct.
  bool applyToParallel = true, applyToTeams = true;

  auto hasParallel = findDirective(llvm::omp::Directive::OMPD_parallel);
  if (hasParallel) {
    auto exclusions = llvm::concat<const llvm::omp::Directive>(
        getWorksharingLoop(), llvm::ArrayRef{
                                  llvm::omp::Directive::OMPD_loop,
                                  llvm::omp::Directive::OMPD_sections,
                                  llvm::omp::Directive::OMPD_taskloop,
                              });
    auto present = [&](llvm::omp::Directive id) {
      return findDirective(id) != nullptr;
    };

    if (llvm::any_of(exclusions, present))
      applyToParallel = false;
  }

  auto hasTeams = findDirective(llvm::omp::Directive::OMPD_teams);
  if (hasTeams) {
    // The only exclusion is OMPD_loop.
    if (findDirective(llvm::omp::Directive::OMPD_loop))
      applyToTeams = false;
  }

  auto &objects = std::get<tomp::ObjectListT<I, E>>(clause.t);

  tomp::ObjectListT<I, E> sharedObjects;
  llvm::transform(objects, std::back_inserter(sharedObjects),
                  [&](const ObjectTy &object) {
                    auto maybeBase = helper.getBaseObject(object);
                    return maybeBase ? *maybeBase : object;
                  });

  // S For the parallel and teams constructs above, the effect of the
  // S reduction clause instead is as if each list item or, for any list
  // S item that is an array item, its corresponding base array or base
  // S pointer appears in a shared clause for the construct.
  if (!sharedObjects.empty()) {
    if (hasParallel && !applyToParallel) {
      auto *shared =
          makeClause(llvm::omp::Clause::OMPC_shared,
                     tomp::clause::SharedT<T, I, E>{/*List=*/sharedObjects});
      hasParallel->clauses.push_back(shared);
    }
    if (hasTeams && !applyToTeams) {
      auto *shared =
          makeClause(llvm::omp::Clause::OMPC_shared,
                     tomp::clause::SharedT<T, I, E>{/*List=*/sharedObjects});
      hasTeams->clauses.push_back(shared);
    }
  }

  // TODO(not implemented in parser yet): Apply the following.
  // S If the task reduction-modifier is specified, the effect is as if
  // S it only modifies the behavior of the reduction clause on the innermost
  // S leaf construct that accepts the modifier (see Section 5.5.8). If the
  // S inscan reduction-modifier is specified, the effect is as if it modifies
  // S the behavior of the reduction clause on all constructs of the combined
  // S construct to which the clause is applied and that accept the modifier.

  bool applied = applyIf(node, [&](DirectiveInfo<T, I, E> &dir) {
    if (!applyToParallel && &dir == hasParallel)
      return false;
    if (!applyToTeams && &dir == hasTeams)
      return false;
    return true;
  });

  // S If a list item in a reduction clause on a combined target construct
  // S does not have the same base variable or base pointer as a list item
  // S in a map clause on the construct, then the effect is as if the list
  // S item in the reduction clause appears as a list item in a map clause
  // S with a map-type of tofrom.
  auto hasTarget = findDirective(llvm::omp::Directive::OMPD_target);
  if (hasTarget && leafs.size() > 1) {
    tomp::ObjectListT<I, E> tofrom;
    llvm::copy_if(objects, std::back_inserter(tofrom),
                  [&](const ObjectTy &object) {
                    if (auto maybeBase = helper.getBaseObject(object))
                      return !mapBases.contains(maybeBase->id());
                    return !mapBases.contains(object.id()); // XXX is this ok?
                  });
    if (!tofrom.empty()) {
      using MapType = typename tomp::clause::MapT<T, I, E>::MapType;
      auto *map = makeClause(
          llvm::omp::Clause::OMPC_map,
          tomp::clause::MapT<T, I, E>{
              {/*MapType=*/MapType::Tofrom, /*MapTypeModifier=*/std::nullopt,
               /*Mapper=*/std::nullopt, /*Iterator=*/std::nullopt,
               /*LocatorList=*/std::move(tofrom)}});

      hasTarget->clauses.push_back(map);
      applied = true;
    }
  }

  return applied;
}

// IF
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::IfT<T, I, E> &clause, const ClauseTy *node) {
  using DirectiveNameModifier =
      typename clause::IfT<T, I, T>::DirectiveNameModifier;
  auto &modifier = std::get<std::optional<DirectiveNameModifier>>(clause.t);

  if (modifier) {
    llvm::omp::Directive dirId = llvm::omp::Directive::OMPD_unknown;

    switch (*modifier) {
    case llvm::omp::Directive::OMPD_parallel:
    case llvm::omp::Directive::OMPD_simd:
    case llvm::omp::Directive::OMPD_target:
    case llvm::omp::Directive::OMPD_task:
    case llvm::omp::Directive::OMPD_taskloop:
    case llvm::omp::Directive::OMPD_teams:
      break;

    default:
      // llvm::errs() << "Invalid modifier in IF clause\n";
      return false;
    }

    if (auto *hasDir = findDirective(dirId)) {
      hasDir->clauses.push_back(node);
      return true;
    }
    // llvm::errs() << "Directive from modifier not found\n";
    return false;
  }

  if (applyToAll(node))
    return true;

  // llvm::errs() << "Cannot apply IF\n";
  return false;
}

// LINEAR
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::LinearT<T, I, E> &clause, const ClauseTy *node) {
  // S The effect of the linear clause is as if it is applied to the innermost
  // S leaf construct.
  if (applyToInnermost(node)) {
    // llvm::errs() << "Cannot apply LINEAR\n";
    return false;
  }

  // The rest is about SIMD.
  if (!findDirective(llvm::omp::OMPD_simd))
    return true;

  // S Additionally, if the list item is not the iteration variable of a
  // S simd or worksharing-loop SIMD construct, the effect on the outer leaf
  // S constructs is as if the list item was specified in firstprivate and
  // S lastprivate clauses on the combined or composite construct, [...]
  //
  // S If a list item of the linear clause is the iteration variable of a
  // S simd or worksharing-loop SIMD construct and it is not declared in
  // S the construct, the effect on the outer leaf constructs is as if the
  // S list item was specified in a lastprivate clause on the combined or
  // S composite construct [...]

  // It's not clear how an object can be listed in a clause AND be the
  // iteration variable of a construct in which is it declared. If an
  // object is declared in the construct, then the declaration is located
  // after the clause listing it.

  std::optional<ObjectTy> iterVar = helper.getIterVar();
  const auto &objects = std::get<tomp::ObjectListT<I, E>>(clause.t);

  // Lists of objects that will be used to construct FIRSTPRIVATE and
  // LASTPRIVATE clauses.
  tomp::ObjectListT<I, E> first, last;

  for (const ObjectTy &object : objects) {
    last.push_back(object);
    if (iterVar && object.id() != iterVar->id())
      first.push_back(object);
  }

  if (!first.empty()) {
    auto *firstp =
        makeClause(llvm::omp::Clause::OMPC_firstprivate,
                   tomp::clause::FirstprivateT<T, I, E>{/*List=*/first});
    add(firstp); // Appending to the main clause list.
  }
  if (!last.empty()) {
    auto *lastp =
        makeClause(llvm::omp::Clause::OMPC_lastprivate,
                   tomp::clause::LastprivateT<T, I, E>{
                       {/*LastprivateModifier=*/std::nullopt, /*List=*/last}});
    add(lastp); // Appending to the main clause list.
  }
  return true;
}

// NOWAIT
template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::applyClause(
    const tomp::clause::NowaitT<T, I, E> &clause, const ClauseTy *node) {
  if (applyToOutermost(node))
    return true;
  // llvm::errs() << "Cannot apply NOWAIT\n";
  return false;
}

template <typename T, typename I, typename E, typename H>
bool CompositeInfoBase<T, I, E, H>::split() {
  bool success = true;

  for (llvm::omp::Directive leaf : llvm::omp::getLeafConstructs(construct))
    leafs.push_back(DirectiveInfo<T, I, E>{leaf, /*clauses=*/{}});

  for (const ClauseTy *node : nodes)
    addClauseSymsToMap(*node, node);

  // First we need to apply LINEAR, because it can generate additional
  // FIRSTPRIVATE and LASTPRIVATE clauses that apply to the combined/
  // composite construct.
  // Collect them separately, because they may modify the clause list.
  llvm::SmallVector<const ClauseTy *> linears;
  for (const ClauseTy *node : nodes) {
    if (node->id == llvm::omp::Clause::OMPC_linear)
      linears.push_back(node);
  }
  for (const auto *node : linears) {
    success =
        success &&
        applyClause(std::get<tomp::clause::LinearT<T, I, E>>(node->u), node);
  }

  // ALLOCATE clauses need to be applied last since they need to see
  // which directives have data-privatizing clauses.
  auto skip = [](const ClauseTy *node) {
    switch (node->id) {
    case llvm::omp::Clause::OMPC_allocate:
    case llvm::omp::Clause::OMPC_linear:
      return true;
    default:
      return false;
    }
  };

  // Apply (almost) all clauses.
  for (const ClauseTy *node : nodes) {
    if (skip(node))
      continue;
    success =
        success &&
        std::visit([&](auto &&s) { return applyClause(s, node); }, node->u);
  }

  // Apply ALLOCATE.
  for (const ClauseTy *node : nodes) {
    if (node->id != llvm::omp::Clause::OMPC_allocate)
      continue;
    success =
        success &&
        std::visit([&](auto &&s) { return applyClause(s, node); }, node->u);
  }

  return success;
}

} // namespace tomp

#endif // FORTRAN_LOWER_OPENMP_CLAUSET_H

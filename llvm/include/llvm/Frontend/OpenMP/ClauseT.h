//===- ClauseT.h -- clause template definitions ---------------------------===//
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
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#define ENUM(Name, ...) enum class Name { __VA_ARGS__ }
#define OPT(x) std::optional<x>

// A number of OpenMP clauses contain values that come from a given set of
// possibilities. In the IR these are usually represented by enums. Both
// clang and flang use different types for the enums, and the enum elements
// representing the same thing may have different values between clang and
// flang.
// Since the representation below tries to adhere to the spec, and be source
// language agnostic, it defines its own enums, independent from any language
// frontend. As a consequence, when instantiating the templates below,
// frontend-specific enums need to be translated into the representation
// used here. The macros below are intended to assist with the conversion.

// Helper macro for enum-class conversion.
#define CLAUSET_SCOPED_ENUM_MEMBER_CONVERT(Ov, Tv)                             \
  if (v == OtherEnum::Ov) {                                                    \
    return ThisEnum::Tv;                                                       \
  }

// Helper macro for enum (non-class) conversion.
#define CLAUSET_UNSCOPED_ENUM_MEMBER_CONVERT(Ov, Tv)                           \
  if (v == Ov) {                                                               \
    return ThisEnum::Tv;                                                       \
  }

#define CLAUSET_ENUM_CONVERT(func, OtherE, ThisE, Maps)                        \
  auto func = [](OtherE v) -> ThisE {                                          \
    using ThisEnum = ThisE;                                                    \
    using OtherEnum = OtherE;                                                  \
    (void)sizeof(OtherEnum); /*Avoid "unused local typedef" warning*/          \
    Maps;                                                                      \
    llvm_unreachable("Unexpected value in " #OtherE);                          \
  }

// Usage:
//
// Given two enums,
//   enum class Other { o1, o2 };
//   enum class This { t1, t2 };
// generate conversion function "Func : Other -> This" with
//   CLAUSET_ENUM_CONVERT(
//       Func, Other, This,
//       CLAUSET_ENUM_MEMBER_CONVERT(o1, t1)      // <- No comma
//       CLAUSET_ENUM_MEMBER_CONVERT(o2, t2)
//       ...
//   )
//
// Note that the sequence of M(other-value, this-value) is separated
// with _spaces_, not commas.

namespace detail {
// Type trait to determine whether T is a specialization of std::variant.
template <typename T> struct is_variant {
  static constexpr bool value = false;
};

template <typename... Ts> struct is_variant<std::variant<Ts...>> {
  static constexpr bool value = true;
};

template <typename T> constexpr bool is_variant_v = is_variant<T>::value;

// Helper utility to create a type which is a union of two given variants.
template <typename...> struct UnionOfTwo;

template <typename... Types1, typename... Types2>
struct UnionOfTwo<std::variant<Types1...>, std::variant<Types2...>> {
  using type = std::variant<Types1..., Types2...>;
};
} // namespace detail

namespace tomp {
namespace type {

// Helper utility to create a type which is a union of an arbitrary number
// of variants.
template <typename...> struct Union;

template <> struct Union<> {
  // Legal to define, illegal to instantiate.
  using type = std::variant<>;
};

template <typename T, typename... Ts> struct Union<T, Ts...> {
  static_assert(detail::is_variant_v<T>);
  using type =
      typename detail::UnionOfTwo<T, typename Union<Ts...>::type>::type;
};

template <typename T> using ListT = llvm::SmallVector<T, 0>;

// The ObjectT class represents a variable (as defined in the OpenMP spec).
//
// A specialization of ObjectT<Id, Expr> must provide the following definitions:
// {
//    using IdTy = Id;
//    using ExprTy = Expr;
//
//    auto id() const -> IdTy {
//      return the identifier of the object (for use in tests for
//         presence/absence of the object)
//    }
//
//    auto ref() const -> (e.g. const ExprTy&) {
//      return the expression accessing (referencing) the object
//    }
// }
//
// For example, the ObjectT instance created for "var[x+1]" would have
// the `id()` return the identifier for `var`, and the `ref()` return the
// representation of the array-access `var[x+1]`.
//
// The identity of an object must always be present, i.e. it cannot be
// nullptr, std::nullopt, etc. The reference is optional.
//
// Note: the ObjectT template is not defined. Any user of it is expected to
// provide their own specialization that conforms to the above requirements.
template <typename IdType, typename ExprType> struct ObjectT;

template <typename I, typename E> using ObjectListT = ListT<ObjectT<I, E>>;

using DirectiveName = llvm::omp::Directive;

template <typename I, typename E> //
struct DefinedOperatorT {
  struct DefinedOpName {
    using WrapperTrait = std::true_type;
    ObjectT<I, E> v;
  };
  ENUM(IntrinsicOperator, Power, Multiply, Divide, Add, Subtract, Concat, LT,
       LE, EQ, NE, GE, GT, NOT, AND, OR, EQV, NEQV, Min, Max);
  using UnionTrait = std::true_type;
  std::variant<DefinedOpName, IntrinsicOperator> u;
};

template <typename E> //
struct RangeT {
  // range-specification: begin : end[: step]
  using TupleTrait = std::true_type;
  std::tuple<E, E, OPT(E)> t;
};

template <typename TypeType, typename IdType, typename ExprType> //
struct IteratorSpecifierT {
  // iterators-specifier: [ iterator-type ] identifier = range-specification
  using TupleTrait = std::true_type;
  std::tuple<OPT(TypeType), ObjectT<IdType, ExprType>, RangeT<ExprType>> t;
};

// Note:
// For motion or map clauses the OpenMP spec allows a unique mapper modifier.
// In practice, since these clauses apply to multiple objects, there can be
// multiple effective mappers applicable to these objects (due to overloads,
// etc.). Because of that store a list of mappers every time a mapper modifier
// is allowed. If the mapper list contains a single element, it applies to
// all objects in the clause, otherwise there should be as many mappers as
// there are objects.
template <typename I, typename E> //
struct MapperT {
  using MapperIdentifier = ObjectT<I, E>;
  using WrapperTrait = std::true_type;
  MapperIdentifier v;
};

ENUM(MemoryOrder, AcqRel, Acquire, Relaxed, Release, SeqCst);
ENUM(MotionExpectation, Present);
ENUM(TaskDependenceType, In, Out, Inout, Mutexinoutset, Inoutset, Depobj);

template <typename I, typename E> //
struct LoopIterationT {
  struct Distance {
    using TupleTrait = std::true_type;
    std::tuple<DefinedOperatorT<I, E>, E> t;
  };
  using TupleTrait = std::true_type;
  std::tuple<ObjectT<I, E>, OPT(Distance)> t;
};

template <typename I, typename E> //
struct ProcedureDesignatorT {
  using WrapperTrait = std::true_type;
  ObjectT<I, E> v;
};

// Note:
// For reduction clauses the OpenMP spec allows a unique reduction identifier.
// For reasons analogous to those listed for the MapperT type, clauses that
// according to the spec contain a reduction identifier will contain a list of
// reduction identifiers. The same constraints apply: there is either a single
// identifier that applies to all objects, or there are as many identifiers
// as there are objects.
template <typename I, typename E> //
struct ReductionIdentifierT {
  using UnionTrait = std::true_type;
  std::variant<DefinedOperatorT<I, E>, ProcedureDesignatorT<I, E>> u;
};

template <typename T, typename I, typename E> //
using IteratorT = ListT<IteratorSpecifierT<T, I, E>>;
} // namespace type

template <typename T> using ListT = type::ListT<T>;

template <typename I, typename E> using ObjectT = type::ObjectT<I, E>;
template <typename I, typename E> using ObjectListT = type::ObjectListT<I, E>;

template <typename T, typename I, typename E>
using IteratorT = type::IteratorT<T, I, E>;

template <
    typename ContainerTy, typename FunctionTy,
    typename ElemTy = typename llvm::remove_cvref_t<ContainerTy>::value_type,
    typename ResultTy = std::invoke_result_t<FunctionTy, ElemTy>>
ListT<ResultTy> makeList(ContainerTy &&container, FunctionTy &&func) {
  ListT<ResultTy> v;
  llvm::transform(container, std::back_inserter(v), func);
  return v;
}

namespace clause {
template <typename T, typename I, typename E> //
struct AbsentT {
  using List = ListT<type::DirectiveName>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct AcqRelT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct AcquireT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct AdjustArgsT {
  using IncompleteTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct AffinityT {
  using Iterator = type::IteratorT<T, I, E>;
  using LocatorList = ObjectListT<I, E>;

  using TupleTrait = std::true_type;
  std::tuple<OPT(Iterator), LocatorList> t;
};

template <typename T, typename I, typename E> //
struct AlignT {
  using Alignment = E;

  using WrapperTrait = std::true_type;
  Alignment v;
};

template <typename T, typename I, typename E> //
struct AlignedT {
  using Alignment = E;
  using List = ObjectListT<I, E>;

  using TupleTrait = std::true_type;
  std::tuple<OPT(Alignment), List> t;
};

template <typename T, typename I, typename E> //
struct AllocatorT;

template <typename T, typename I, typename E> //
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

template <typename T, typename I, typename E> //
struct AllocatorT {
  using Allocator = E;
  using WrapperTrait = std::true_type;
  Allocator v;
};

template <typename T, typename I, typename E> //
struct AppendArgsT {
  using IncompleteTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct AtT {
  ENUM(ActionTime, Compilation, Execution);
  using WrapperTrait = std::true_type;
  ActionTime v;
};

template <typename T, typename I, typename E> //
struct AtomicDefaultMemOrderT {
  using MemoryOrder = type::MemoryOrder;
  using WrapperTrait = std::true_type;
  MemoryOrder v; // Name not provided in spec
};

template <typename T, typename I, typename E> //
struct BindT {
  ENUM(Binding, Teams, Parallel, Thread);
  using WrapperTrait = std::true_type;
  Binding v;
};

template <typename T, typename I, typename E> //
struct CaptureT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct CollapseT {
  using N = E;
  using WrapperTrait = std::true_type;
  N v;
};

template <typename T, typename I, typename E> //
struct CompareT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct ContainsT {
  using List = ListT<type::DirectiveName>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct CopyinT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct CopyprivateT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct DefaultT {
  ENUM(DataSharingAttribute, Firstprivate, None, Private, Shared);
  using WrapperTrait = std::true_type;
  DataSharingAttribute v;
};

template <typename T, typename I, typename E> //
struct DefaultmapT {
  ENUM(ImplicitBehavior, Alloc, To, From, Tofrom, Firstprivate, None, Default,
       Present);
  ENUM(VariableCategory, Scalar, Aggregate, Pointer, Allocatable);
  using TupleTrait = std::true_type;
  std::tuple<ImplicitBehavior, OPT(VariableCategory)> t;
};

template <typename T, typename I, typename E> //
struct DoacrossT;

template <typename T, typename I, typename E> //
struct DependT {
  using Iterator = type::IteratorT<T, I, E>;
  using LocatorList = ObjectListT<I, E>;
  using TaskDependenceType = tomp::type::TaskDependenceType;

  struct WithLocators { // Modern form
    using TupleTrait = std::true_type;
    // Empty LocatorList means "omp_all_memory".
    std::tuple<TaskDependenceType, OPT(Iterator), LocatorList> t;
  };

  using Doacross = DoacrossT<T, I, E>;
  using UnionTrait = std::true_type;
  std::variant<Doacross, WithLocators> u; // Doacross form is legacy
};

template <typename T, typename I, typename E> //
struct DestroyT {
  using DestroyVar = ObjectT<I, E>;
  using WrapperTrait = std::true_type;
  // DestroyVar can be ommitted in "depobj destroy".
  OPT(DestroyVar) v;
};

template <typename T, typename I, typename E> //
struct DetachT {
  using EventHandle = ObjectT<I, E>;
  using WrapperTrait = std::true_type;
  EventHandle v;
};

template <typename T, typename I, typename E> //
struct DeviceT {
  using DeviceDescription = E;
  ENUM(DeviceModifier, Ancestor, DeviceNum);
  using TupleTrait = std::true_type;
  std::tuple<OPT(DeviceModifier), DeviceDescription> t;
};

template <typename T, typename I, typename E> //
struct DeviceTypeT {
  ENUM(DeviceTypeDescription, Any, Host, Nohost);
  using WrapperTrait = std::true_type;
  DeviceTypeDescription v;
};

template <typename T, typename I, typename E> //
struct DistScheduleT {
  ENUM(Kind, Static);
  using ChunkSize = E;
  using TupleTrait = std::true_type;
  std::tuple<Kind, OPT(ChunkSize)> t;
};

template <typename T, typename I, typename E> //
struct DoacrossT {
  using Vector = ListT<type::LoopIterationT<I, E>>;
  ENUM(DependenceType, Source, Sink);
  using TupleTrait = std::true_type;
  // Empty Vector means "omp_cur_iteration"
  std::tuple<DependenceType, Vector> t;
};

template <typename T, typename I, typename E> //
struct DynamicAllocatorsT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct EnterT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct ExclusiveT {
  using WrapperTrait = std::true_type;
  using List = ObjectListT<I, E>;
  List v;
};

template <typename T, typename I, typename E> //
struct FailT {
  using MemoryOrder = type::MemoryOrder;
  using WrapperTrait = std::true_type;
  MemoryOrder v;
};

template <typename T, typename I, typename E> //
struct FilterT {
  using ThreadNum = E;
  using WrapperTrait = std::true_type;
  ThreadNum v;
};

template <typename T, typename I, typename E> //
struct FinalT {
  using Finalize = E;
  using WrapperTrait = std::true_type;
  Finalize v;
};

template <typename T, typename I, typename E> //
struct FirstprivateT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct FromT {
  using LocatorList = ObjectListT<I, E>;
  using Expectation = type::MotionExpectation;
  using Iterator = type::IteratorT<T, I, E>;
  // See note at the definition of the MapperT type.
  using Mappers = ListT<type::MapperT<I, E>>; // Not a spec name

  using TupleTrait = std::true_type;
  std::tuple<OPT(Expectation), OPT(Mappers), OPT(Iterator), LocatorList> t;
};

template <typename T, typename I, typename E> //
struct FullT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct GrainsizeT {
  ENUM(Prescriptiveness, Strict);
  using GrainSize = E;
  using TupleTrait = std::true_type;
  std::tuple<OPT(Prescriptiveness), GrainSize> t;
};

template <typename T, typename I, typename E> //
struct HasDeviceAddrT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct HintT {
  using HintExpr = E;
  using WrapperTrait = std::true_type;
  HintExpr v;
};

template <typename T, typename I, typename E> //
struct HoldsT {
  using WrapperTrait = std::true_type;
  E v; // No argument name in spec 5.2
};

template <typename T, typename I, typename E> //
struct IfT {
  using DirectiveNameModifier = type::DirectiveName;
  using IfExpression = E;
  using TupleTrait = std::true_type;
  std::tuple<OPT(DirectiveNameModifier), IfExpression> t;
};

template <typename T, typename I, typename E> //
struct InbranchT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct InclusiveT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct IndirectT {
  using InvokedByFptr = E;
  using WrapperTrait = std::true_type;
  InvokedByFptr v;
};

template <typename T, typename I, typename E> //
struct InitT {
  using ForeignRuntimeId = E;
  using InteropVar = ObjectT<I, E>;
  using InteropPreference = ListT<ForeignRuntimeId>;
  ENUM(InteropType, Target, Targetsync);   // Repeatable
  using InteropTypes = ListT<InteropType>; // Not a spec name

  using TupleTrait = std::true_type;
  std::tuple<OPT(InteropPreference), InteropTypes, InteropVar> t;
};

template <typename T, typename I, typename E> //
struct InitializerT {
  using InitializerExpr = E;
  using WrapperTrait = std::true_type;
  InitializerExpr v;
};

template <typename T, typename I, typename E> //
struct InReductionT {
  using List = ObjectListT<I, E>;
  // See note at the definition of the ReductionIdentifierT type.
  // The name ReductionIdentifiers is not a spec name.
  using ReductionIdentifiers = ListT<type::ReductionIdentifierT<I, E>>;
  using TupleTrait = std::true_type;
  std::tuple<ReductionIdentifiers, List> t;
};

template <typename T, typename I, typename E> //
struct IsDevicePtrT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct LastprivateT {
  using List = ObjectListT<I, E>;
  ENUM(LastprivateModifier, Conditional);
  using TupleTrait = std::true_type;
  std::tuple<OPT(LastprivateModifier), List> t;
};

template <typename T, typename I, typename E> //
struct LinearT {
  // std::get<type> won't work here due to duplicate types in the tuple.
  using List = ObjectListT<I, E>;
  using StepSimpleModifier = E;
  using StepComplexModifier = E;
  ENUM(LinearModifier, Ref, Val, Uval);

  using TupleTrait = std::true_type;
  // Step == nullptr means 1.
  std::tuple<OPT(StepSimpleModifier), OPT(StepComplexModifier),
             OPT(LinearModifier), List>
      t;
};

template <typename T, typename I, typename E> //
struct LinkT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct MapT {
  using LocatorList = ObjectListT<I, E>;
  ENUM(MapType, To, From, Tofrom, Alloc, Release, Delete);
  ENUM(MapTypeModifier, Always, Close, Present, OmpxHold);
  // See note at the definition of the MapperT type.
  using Mappers = ListT<type::MapperT<I, E>>; // Not a spec name
  using Iterator = type::IteratorT<T, I, E>;
  using MapTypeModifiers = ListT<MapTypeModifier>; // Not a spec name

  using TupleTrait = std::true_type;
  std::tuple<OPT(MapType), OPT(MapTypeModifiers), OPT(Mappers), OPT(Iterator),
             LocatorList>
      t;
};

template <typename T, typename I, typename E> //
struct MatchT {
  using IncompleteTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct MergeableT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct MessageT {
  using MsgString = E;
  using WrapperTrait = std::true_type;
  MsgString v;
};

template <typename T, typename I, typename E> //
struct NocontextT {
  using DoNotUpdateContext = E;
  using WrapperTrait = std::true_type;
  DoNotUpdateContext v;
};

template <typename T, typename I, typename E> //
struct NogroupT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct NontemporalT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct NoOpenmpT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct NoOpenmpRoutinesT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct NoParallelismT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct NotinbranchT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct NovariantsT {
  using DoNotUseVariant = E;
  using WrapperTrait = std::true_type;
  DoNotUseVariant v;
};

template <typename T, typename I, typename E> //
struct NowaitT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct NumTasksT {
  using NumTasks = E;
  ENUM(Prescriptiveness, Strict);
  using TupleTrait = std::true_type;
  std::tuple<OPT(Prescriptiveness), NumTasks> t;
};

template <typename T, typename I, typename E> //
struct NumTeamsT {
  using TupleTrait = std::true_type;
  using LowerBound = E;
  using UpperBound = E;
  std::tuple<OPT(LowerBound), UpperBound> t;
};

template <typename T, typename I, typename E> //
struct NumThreadsT {
  using Nthreads = E;
  using WrapperTrait = std::true_type;
  Nthreads v;
};

template <typename T, typename I, typename E> //
struct OmpxAttributeT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct OmpxBareT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct OmpxDynCgroupMemT {
  using WrapperTrait = std::true_type;
  E v;
};

template <typename T, typename I, typename E> //
struct OrderT {
  ENUM(OrderModifier, Reproducible, Unconstrained);
  ENUM(Ordering, Concurrent);
  using TupleTrait = std::true_type;
  std::tuple<OPT(OrderModifier), Ordering> t;
};

template <typename T, typename I, typename E> //
struct OrderedT {
  using N = E;
  using WrapperTrait = std::true_type;
  OPT(N) v;
};

template <typename T, typename I, typename E> //
struct OtherwiseT {
  using IncompleteTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct PartialT {
  using UnrollFactor = E;
  using WrapperTrait = std::true_type;
  OPT(UnrollFactor) v;
};

template <typename T, typename I, typename E> //
struct PriorityT {
  using PriorityValue = E;
  using WrapperTrait = std::true_type;
  PriorityValue v;
};

template <typename T, typename I, typename E> //
struct PrivateT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct ProcBindT {
  ENUM(AffinityPolicy, Close, Master, Spread, Primary);
  using WrapperTrait = std::true_type;
  AffinityPolicy v;
};

template <typename T, typename I, typename E> //
struct ReadT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct ReductionT {
  using List = ObjectListT<I, E>;
  // See note at the definition of the ReductionIdentifierT type.
  // The name ReductionIdentifiers is not a spec name.
  using ReductionIdentifiers = ListT<type::ReductionIdentifierT<I, E>>;
  ENUM(ReductionModifier, Default, Inscan, Task);
  using TupleTrait = std::true_type;
  std::tuple<ReductionIdentifiers, OPT(ReductionModifier), List> t;
};

template <typename T, typename I, typename E> //
struct RelaxedT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct ReleaseT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct ReverseOffloadT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct SafelenT {
  using Length = E;
  using WrapperTrait = std::true_type;
  Length v;
};

template <typename T, typename I, typename E> //
struct ScheduleT {
  ENUM(Kind, Static, Dynamic, Guided, Auto, Runtime);
  using ChunkSize = E;
  ENUM(OrderingModifier, Monotonic, Nonmonotonic);
  ENUM(ChunkModifier, Simd);
  using TupleTrait = std::true_type;
  std::tuple<Kind, OPT(OrderingModifier), OPT(ChunkModifier), OPT(ChunkSize)> t;
};

template <typename T, typename I, typename E> //
struct SeqCstT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct SeverityT {
  ENUM(SevLevel, Fatal, Warning);
  using WrapperTrait = std::true_type;
  SevLevel v;
};

template <typename T, typename I, typename E> //
struct SharedT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct SimdT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct SimdlenT {
  using Length = E;
  using WrapperTrait = std::true_type;
  Length v;
};

template <typename T, typename I, typename E> //
struct SizesT {
  using SizeList = ListT<E>;
  using WrapperTrait = std::true_type;
  SizeList v;
};

template <typename T, typename I, typename E> //
struct TaskReductionT {
  using List = ObjectListT<I, E>;
  // See note at the definition of the ReductionIdentifierT type.
  // The name ReductionIdentifiers is not a spec name.
  using ReductionIdentifiers = ListT<type::ReductionIdentifierT<I, E>>;
  using TupleTrait = std::true_type;
  std::tuple<ReductionIdentifiers, List> t;
};

template <typename T, typename I, typename E> //
struct ThreadLimitT {
  using Threadlim = E;
  using WrapperTrait = std::true_type;
  Threadlim v;
};

template <typename T, typename I, typename E> //
struct ThreadsT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct ToT {
  using LocatorList = ObjectListT<I, E>;
  using Expectation = type::MotionExpectation;
  // See note at the definition of the MapperT type.
  using Mappers = ListT<type::MapperT<I, E>>; // Not a spec name
  using Iterator = type::IteratorT<T, I, E>;

  using TupleTrait = std::true_type;
  std::tuple<OPT(Expectation), OPT(Mappers), OPT(Iterator), LocatorList> t;
};

template <typename T, typename I, typename E> //
struct UnifiedAddressT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct UnifiedSharedMemoryT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct UniformT {
  using ParameterList = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  ParameterList v;
};

template <typename T, typename I, typename E> //
struct UnknownT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct UntiedT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct UpdateT {
  using TaskDependenceType = tomp::type::TaskDependenceType;
  using WrapperTrait = std::true_type;
  OPT(TaskDependenceType) v;
};

template <typename T, typename I, typename E> //
struct UseT {
  using InteropVar = ObjectT<I, E>;
  using WrapperTrait = std::true_type;
  InteropVar v;
};

template <typename T, typename I, typename E> //
struct UseDeviceAddrT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct UseDevicePtrT {
  using List = ObjectListT<I, E>;
  using WrapperTrait = std::true_type;
  List v;
};

template <typename T, typename I, typename E> //
struct UsesAllocatorsT {
  using MemSpace = E;
  using TraitsArray = ObjectT<I, E>;
  using Allocator = E;
  using AllocatorSpec =
      std::tuple<OPT(MemSpace), OPT(TraitsArray), Allocator>; // Not a spec name
  using Allocators = ListT<AllocatorSpec>;                    // Not a spec name
  using WrapperTrait = std::true_type;
  Allocators v;
};

template <typename T, typename I, typename E> //
struct WeakT {
  using EmptyTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct WhenT {
  using IncompleteTrait = std::true_type;
};

template <typename T, typename I, typename E> //
struct WriteT {
  using EmptyTrait = std::true_type;
};

// ---

template <typename T, typename I, typename E>
using ExtensionClausesT =
    std::variant<OmpxAttributeT<T, I, E>, OmpxBareT<T, I, E>,
                 OmpxDynCgroupMemT<T, I, E>>;

template <typename T, typename I, typename E>
using EmptyClausesT = std::variant<
    AcqRelT<T, I, E>, AcquireT<T, I, E>, CaptureT<T, I, E>, CompareT<T, I, E>,
    DynamicAllocatorsT<T, I, E>, FullT<T, I, E>, InbranchT<T, I, E>,
    MergeableT<T, I, E>, NogroupT<T, I, E>, NoOpenmpRoutinesT<T, I, E>,
    NoOpenmpT<T, I, E>, NoParallelismT<T, I, E>, NotinbranchT<T, I, E>,
    NowaitT<T, I, E>, ReadT<T, I, E>, RelaxedT<T, I, E>, ReleaseT<T, I, E>,
    ReverseOffloadT<T, I, E>, SeqCstT<T, I, E>, SimdT<T, I, E>,
    ThreadsT<T, I, E>, UnifiedAddressT<T, I, E>, UnifiedSharedMemoryT<T, I, E>,
    UnknownT<T, I, E>, UntiedT<T, I, E>, UseT<T, I, E>, WeakT<T, I, E>,
    WriteT<T, I, E>>;

template <typename T, typename I, typename E>
using IncompleteClausesT =
    std::variant<AdjustArgsT<T, I, E>, AppendArgsT<T, I, E>, MatchT<T, I, E>,
                 OtherwiseT<T, I, E>, WhenT<T, I, E>>;

template <typename T, typename I, typename E>
using TupleClausesT =
    std::variant<AffinityT<T, I, E>, AlignedT<T, I, E>, AllocateT<T, I, E>,
                 DefaultmapT<T, I, E>, DeviceT<T, I, E>, DistScheduleT<T, I, E>,
                 DoacrossT<T, I, E>, FromT<T, I, E>, GrainsizeT<T, I, E>,
                 IfT<T, I, E>, InitT<T, I, E>, InReductionT<T, I, E>,
                 LastprivateT<T, I, E>, LinearT<T, I, E>, MapT<T, I, E>,
                 NumTasksT<T, I, E>, OrderT<T, I, E>, ReductionT<T, I, E>,
                 ScheduleT<T, I, E>, TaskReductionT<T, I, E>, ToT<T, I, E>>;

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
    FirstprivateT<T, I, E>, HasDeviceAddrT<T, I, E>, HintT<T, I, E>,
    HoldsT<T, I, E>, InclusiveT<T, I, E>, IndirectT<T, I, E>,
    InitializerT<T, I, E>, IsDevicePtrT<T, I, E>, LinkT<T, I, E>,
    MessageT<T, I, E>, NocontextT<T, I, E>, NontemporalT<T, I, E>,
    NovariantsT<T, I, E>, NumTeamsT<T, I, E>, NumThreadsT<T, I, E>,
    OrderedT<T, I, E>, PartialT<T, I, E>, PriorityT<T, I, E>, PrivateT<T, I, E>,
    ProcBindT<T, I, E>, SafelenT<T, I, E>, SeverityT<T, I, E>, SharedT<T, I, E>,
    SimdlenT<T, I, E>, SizesT<T, I, E>, ThreadLimitT<T, I, E>,
    UniformT<T, I, E>, UpdateT<T, I, E>, UseDeviceAddrT<T, I, E>,
    UseDevicePtrT<T, I, E>, UsesAllocatorsT<T, I, E>>;

template <typename T, typename I, typename E>
using UnionOfAllClausesT = typename type::Union< //
    EmptyClausesT<T, I, E>,                      //
    ExtensionClausesT<T, I, E>,                  //
    IncompleteClausesT<T, I, E>,                 //
    TupleClausesT<T, I, E>,                      //
    UnionClausesT<T, I, E>,                      //
    WrapperClausesT<T, I, E>                     //
    >::type;

} // namespace clause

// The variant wrapper that encapsulates all possible specific clauses.
// The `Extras` arguments are additional types representing local extensions
// to the clause set, e.g.
//
// using Clause = ClauseT<Type, Id, Expr,
//                        MyClause1, MyClause2>;
//
// The member Clause::u will be a variant containing all specific clauses
// defined above, plus MyClause1 and MyClause2.
template <typename TypeType, typename IdType, typename ExprType,
          typename... Extras>
struct ClauseT {
  using TypeTy = TypeType;
  using IdTy = IdType;
  using ExprTy = ExprType;

  using VariantTy = typename type::Union<
      clause::UnionOfAllClausesT<TypeType, IdType, ExprType>,
      std::variant<Extras...>>::type;

  llvm::omp::Clause id; // The numeric id of the clause
  using UnionTrait = std::true_type;
  VariantTy u;
};

} // namespace tomp

#undef OPT
#undef ENUM

#endif // FORTRAN_LOWER_OPENMP_CLAUSET_H

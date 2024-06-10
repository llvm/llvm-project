//===- llvm/unittests/Frontend/OpenMPDecompositionTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenMP/ClauseT.h"
#include "llvm/Frontend/OpenMP/ConstructDecompositionT.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "gtest/gtest.h"

#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

// The actual tests start at comment "--- Test" below.

// Create simple instantiations of all clauses to allow manual construction
// of clauses, and implement emitting of a directive with clauses to a string.
//
// The tests then follow the pattern
// 1. Create a list of clauses.
// 2. Pass them, together with a construct, to the decomposition class.
// 3. Extract individual resulting leaf constructs with clauses applied
//    to them.
// 4. Convert them to strings and compare with expected outputs.

namespace omp {
struct TypeTy {}; // placeholder
struct ExprTy {}; // placeholder
using IdTy = std::string;
} // namespace omp

namespace tomp::type {
template <> struct ObjectT<omp::IdTy, omp::ExprTy> {
  const omp::IdTy &id() const { return name; }
  const std::optional<omp::ExprTy> ref() const { return omp::ExprTy{}; }

  omp::IdTy name;
};
} // namespace tomp::type

namespace omp {
template <typename ElemTy> using List = tomp::type::ListT<ElemTy>;

using Object = tomp::ObjectT<IdTy, ExprTy>;

namespace clause {
using DefinedOperator = tomp::type::DefinedOperatorT<IdTy, ExprTy>;
using ProcedureDesignator = tomp::type::ProcedureDesignatorT<IdTy, ExprTy>;
using ReductionOperator = tomp::type::ReductionIdentifierT<IdTy, ExprTy>;

using AcqRel = tomp::clause::AcqRelT<TypeTy, IdTy, ExprTy>;
using Acquire = tomp::clause::AcquireT<TypeTy, IdTy, ExprTy>;
using AdjustArgs = tomp::clause::AdjustArgsT<TypeTy, IdTy, ExprTy>;
using Affinity = tomp::clause::AffinityT<TypeTy, IdTy, ExprTy>;
using Aligned = tomp::clause::AlignedT<TypeTy, IdTy, ExprTy>;
using Align = tomp::clause::AlignT<TypeTy, IdTy, ExprTy>;
using Allocate = tomp::clause::AllocateT<TypeTy, IdTy, ExprTy>;
using Allocator = tomp::clause::AllocatorT<TypeTy, IdTy, ExprTy>;
using AppendArgs = tomp::clause::AppendArgsT<TypeTy, IdTy, ExprTy>;
using AtomicDefaultMemOrder =
    tomp::clause::AtomicDefaultMemOrderT<TypeTy, IdTy, ExprTy>;
using At = tomp::clause::AtT<TypeTy, IdTy, ExprTy>;
using Bind = tomp::clause::BindT<TypeTy, IdTy, ExprTy>;
using Capture = tomp::clause::CaptureT<TypeTy, IdTy, ExprTy>;
using Collapse = tomp::clause::CollapseT<TypeTy, IdTy, ExprTy>;
using Compare = tomp::clause::CompareT<TypeTy, IdTy, ExprTy>;
using Copyin = tomp::clause::CopyinT<TypeTy, IdTy, ExprTy>;
using Copyprivate = tomp::clause::CopyprivateT<TypeTy, IdTy, ExprTy>;
using Defaultmap = tomp::clause::DefaultmapT<TypeTy, IdTy, ExprTy>;
using Default = tomp::clause::DefaultT<TypeTy, IdTy, ExprTy>;
using Depend = tomp::clause::DependT<TypeTy, IdTy, ExprTy>;
using Destroy = tomp::clause::DestroyT<TypeTy, IdTy, ExprTy>;
using Detach = tomp::clause::DetachT<TypeTy, IdTy, ExprTy>;
using Device = tomp::clause::DeviceT<TypeTy, IdTy, ExprTy>;
using DeviceType = tomp::clause::DeviceTypeT<TypeTy, IdTy, ExprTy>;
using DistSchedule = tomp::clause::DistScheduleT<TypeTy, IdTy, ExprTy>;
using Doacross = tomp::clause::DoacrossT<TypeTy, IdTy, ExprTy>;
using DynamicAllocators =
    tomp::clause::DynamicAllocatorsT<TypeTy, IdTy, ExprTy>;
using Enter = tomp::clause::EnterT<TypeTy, IdTy, ExprTy>;
using Exclusive = tomp::clause::ExclusiveT<TypeTy, IdTy, ExprTy>;
using Fail = tomp::clause::FailT<TypeTy, IdTy, ExprTy>;
using Filter = tomp::clause::FilterT<TypeTy, IdTy, ExprTy>;
using Final = tomp::clause::FinalT<TypeTy, IdTy, ExprTy>;
using Firstprivate = tomp::clause::FirstprivateT<TypeTy, IdTy, ExprTy>;
using From = tomp::clause::FromT<TypeTy, IdTy, ExprTy>;
using Full = tomp::clause::FullT<TypeTy, IdTy, ExprTy>;
using Grainsize = tomp::clause::GrainsizeT<TypeTy, IdTy, ExprTy>;
using HasDeviceAddr = tomp::clause::HasDeviceAddrT<TypeTy, IdTy, ExprTy>;
using Hint = tomp::clause::HintT<TypeTy, IdTy, ExprTy>;
using If = tomp::clause::IfT<TypeTy, IdTy, ExprTy>;
using Inbranch = tomp::clause::InbranchT<TypeTy, IdTy, ExprTy>;
using Inclusive = tomp::clause::InclusiveT<TypeTy, IdTy, ExprTy>;
using Indirect = tomp::clause::IndirectT<TypeTy, IdTy, ExprTy>;
using Init = tomp::clause::InitT<TypeTy, IdTy, ExprTy>;
using InReduction = tomp::clause::InReductionT<TypeTy, IdTy, ExprTy>;
using IsDevicePtr = tomp::clause::IsDevicePtrT<TypeTy, IdTy, ExprTy>;
using Lastprivate = tomp::clause::LastprivateT<TypeTy, IdTy, ExprTy>;
using Linear = tomp::clause::LinearT<TypeTy, IdTy, ExprTy>;
using Link = tomp::clause::LinkT<TypeTy, IdTy, ExprTy>;
using Map = tomp::clause::MapT<TypeTy, IdTy, ExprTy>;
using Match = tomp::clause::MatchT<TypeTy, IdTy, ExprTy>;
using Mergeable = tomp::clause::MergeableT<TypeTy, IdTy, ExprTy>;
using Message = tomp::clause::MessageT<TypeTy, IdTy, ExprTy>;
using Nocontext = tomp::clause::NocontextT<TypeTy, IdTy, ExprTy>;
using Nogroup = tomp::clause::NogroupT<TypeTy, IdTy, ExprTy>;
using Nontemporal = tomp::clause::NontemporalT<TypeTy, IdTy, ExprTy>;
using Notinbranch = tomp::clause::NotinbranchT<TypeTy, IdTy, ExprTy>;
using Novariants = tomp::clause::NovariantsT<TypeTy, IdTy, ExprTy>;
using Nowait = tomp::clause::NowaitT<TypeTy, IdTy, ExprTy>;
using NumTasks = tomp::clause::NumTasksT<TypeTy, IdTy, ExprTy>;
using NumTeams = tomp::clause::NumTeamsT<TypeTy, IdTy, ExprTy>;
using NumThreads = tomp::clause::NumThreadsT<TypeTy, IdTy, ExprTy>;
using OmpxAttribute = tomp::clause::OmpxAttributeT<TypeTy, IdTy, ExprTy>;
using OmpxBare = tomp::clause::OmpxBareT<TypeTy, IdTy, ExprTy>;
using OmpxDynCgroupMem = tomp::clause::OmpxDynCgroupMemT<TypeTy, IdTy, ExprTy>;
using Ordered = tomp::clause::OrderedT<TypeTy, IdTy, ExprTy>;
using Order = tomp::clause::OrderT<TypeTy, IdTy, ExprTy>;
using Partial = tomp::clause::PartialT<TypeTy, IdTy, ExprTy>;
using Priority = tomp::clause::PriorityT<TypeTy, IdTy, ExprTy>;
using Private = tomp::clause::PrivateT<TypeTy, IdTy, ExprTy>;
using ProcBind = tomp::clause::ProcBindT<TypeTy, IdTy, ExprTy>;
using Read = tomp::clause::ReadT<TypeTy, IdTy, ExprTy>;
using Reduction = tomp::clause::ReductionT<TypeTy, IdTy, ExprTy>;
using Relaxed = tomp::clause::RelaxedT<TypeTy, IdTy, ExprTy>;
using Release = tomp::clause::ReleaseT<TypeTy, IdTy, ExprTy>;
using ReverseOffload = tomp::clause::ReverseOffloadT<TypeTy, IdTy, ExprTy>;
using Safelen = tomp::clause::SafelenT<TypeTy, IdTy, ExprTy>;
using Schedule = tomp::clause::ScheduleT<TypeTy, IdTy, ExprTy>;
using SeqCst = tomp::clause::SeqCstT<TypeTy, IdTy, ExprTy>;
using Severity = tomp::clause::SeverityT<TypeTy, IdTy, ExprTy>;
using Shared = tomp::clause::SharedT<TypeTy, IdTy, ExprTy>;
using Simdlen = tomp::clause::SimdlenT<TypeTy, IdTy, ExprTy>;
using Simd = tomp::clause::SimdT<TypeTy, IdTy, ExprTy>;
using Sizes = tomp::clause::SizesT<TypeTy, IdTy, ExprTy>;
using TaskReduction = tomp::clause::TaskReductionT<TypeTy, IdTy, ExprTy>;
using ThreadLimit = tomp::clause::ThreadLimitT<TypeTy, IdTy, ExprTy>;
using Threads = tomp::clause::ThreadsT<TypeTy, IdTy, ExprTy>;
using To = tomp::clause::ToT<TypeTy, IdTy, ExprTy>;
using UnifiedAddress = tomp::clause::UnifiedAddressT<TypeTy, IdTy, ExprTy>;
using UnifiedSharedMemory =
    tomp::clause::UnifiedSharedMemoryT<TypeTy, IdTy, ExprTy>;
using Uniform = tomp::clause::UniformT<TypeTy, IdTy, ExprTy>;
using Unknown = tomp::clause::UnknownT<TypeTy, IdTy, ExprTy>;
using Untied = tomp::clause::UntiedT<TypeTy, IdTy, ExprTy>;
using Update = tomp::clause::UpdateT<TypeTy, IdTy, ExprTy>;
using UseDeviceAddr = tomp::clause::UseDeviceAddrT<TypeTy, IdTy, ExprTy>;
using UseDevicePtr = tomp::clause::UseDevicePtrT<TypeTy, IdTy, ExprTy>;
using UsesAllocators = tomp::clause::UsesAllocatorsT<TypeTy, IdTy, ExprTy>;
using Use = tomp::clause::UseT<TypeTy, IdTy, ExprTy>;
using Weak = tomp::clause::WeakT<TypeTy, IdTy, ExprTy>;
using When = tomp::clause::WhenT<TypeTy, IdTy, ExprTy>;
using Write = tomp::clause::WriteT<TypeTy, IdTy, ExprTy>;
} // namespace clause

struct Helper {
  std::optional<Object> getBaseObject(const Object &object) {
    return std::nullopt;
  }
  std::optional<Object> getLoopIterVar() { return std::nullopt; }
};

using Clause = tomp::ClauseT<TypeTy, IdTy, ExprTy>;
using ConstructDecomposition = tomp::ConstructDecompositionT<Clause, Helper>;
using DirectiveWithClauses = tomp::DirectiveWithClauses<Clause>;
} // namespace omp

struct StringifyClause {
  static std::string join(const omp::List<std::string> &Strings) {
    std::stringstream Stream;
    for (const auto &[Index, String] : llvm::enumerate(Strings)) {
      if (Index != 0)
        Stream << ", ";
      Stream << String;
    }
    return Stream.str();
  }

  static std::string to_str(llvm::omp::Directive D) {
    return getOpenMPDirectiveName(D).str();
  }
  static std::string to_str(llvm::omp::Clause C) {
    return getOpenMPClauseName(C).str();
  }
  static std::string to_str(const omp::TypeTy &Type) { return "type"; }
  static std::string to_str(const omp::ExprTy &Expr) { return "expr"; }
  static std::string to_str(const omp::Object &Obj) { return Obj.id(); }

  template <typename U>
  static std::enable_if_t<std::is_enum_v<llvm::remove_cvref_t<U>>, std::string>
  to_str(U &&Item) {
    return std::to_string(llvm::to_underlying(Item));
  }

  template <typename U> static std::string to_str(const omp::List<U> &Items) {
    omp::List<std::string> Names;
    llvm::transform(Items, std::back_inserter(Names),
                    [](auto &&S) { return to_str(S); });
    return "(" + join(Names) + ")";
  }

  template <typename U>
  static std::string to_str(const std::optional<U> &Item) {
    if (Item)
      return to_str(*Item);
    return "";
  }

  template <typename... Us, size_t... Is>
  static std::string to_str(const std::tuple<Us...> &Tuple,
                            std::index_sequence<Is...>) {
    omp::List<std::string> Strings;
    (Strings.push_back(to_str(std::get<Is>(Tuple))), ...);
    return "(" + join(Strings) + ")";
  }

  template <typename U>
  static std::enable_if_t<llvm::remove_cvref_t<U>::EmptyTrait::value,
                          std::string>
  to_str(U &&Item) {
    return "";
  }

  template <typename U>
  static std::enable_if_t<llvm::remove_cvref_t<U>::IncompleteTrait::value,
                          std::string>
  to_str(U &&Item) {
    return "";
  }

  template <typename U>
  static std::enable_if_t<llvm::remove_cvref_t<U>::WrapperTrait::value,
                          std::string>
  to_str(U &&Item) {
    // For a wrapper, stringify the wrappee, and only add parentheses if
    // there aren't any already.
    std::string Str = to_str(Item.v);
    if (!Str.empty()) {
      if (Str.front() == '(' && Str.back() == ')')
        return Str;
    }
    return "(" + to_str(Item.v) + ")";
  }

  template <typename U>
  static std::enable_if_t<llvm::remove_cvref_t<U>::TupleTrait::value,
                          std::string>
  to_str(U &&Item) {
    constexpr size_t TupleSize =
        std::tuple_size_v<llvm::remove_cvref_t<decltype(Item.t)>>;
    return to_str(Item.t, std::make_index_sequence<TupleSize>{});
  }

  template <typename U>
  static std::enable_if_t<llvm::remove_cvref_t<U>::UnionTrait::value,
                          std::string>
  to_str(U &&Item) {
    return std::visit([](auto &&S) { return to_str(S); }, Item.u);
  }

  StringifyClause(const omp::Clause &C)
      // Rely on content stringification to emit enclosing parentheses.
      : Str(to_str(C.id) + to_str(C)) {}

  std::string Str;
};

std::string stringify(const omp::DirectiveWithClauses &DWC) {
  std::stringstream Stream;

  Stream << getOpenMPDirectiveName(DWC.id).str();
  for (const omp::Clause &C : DWC.clauses)
    Stream << ' ' << StringifyClause(C).Str;

  return Stream.str();
}

// --- Tests ----------------------------------------------------------

namespace red {
// Make it easier to construct reduction operators from built-in intrinsics.
omp::clause::ReductionOperator
makeOp(omp::clause::DefinedOperator::IntrinsicOperator Op) {
  return omp::clause::ReductionOperator{omp::clause::DefinedOperator{Op}};
}
} // namespace red

namespace {
using namespace llvm::omp;

class OpenMPDecompositionTest : public testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  omp::Helper Helper;
  uint32_t AnyVersion = 999;
};

// PRIVATE
// [5.2:111:5-7]
// Directives: distribute, do, for, loop, parallel, scope, sections, simd,
// single, target, task, taskloop, teams
//
// [5.2:340:1-2]
// (1) The effect of the 1 private clause is as if it is applied only to the
// innermost leaf construct that permits it.
TEST_F(OpenMPDecompositionTest, Private1) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_private, omp::clause::Private{{x}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_parallel_sections,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "parallel");            // (1)
  ASSERT_EQ(Dir1, "sections private(x)"); // (1)
}

TEST_F(OpenMPDecompositionTest, Private2) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_private, omp::clause::Private{{x}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_parallel_masked,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "parallel private(x)"); // (1)
  ASSERT_EQ(Dir1, "masked");              // (1)
}

// FIRSTPRIVATE
// [5.2:112:5-7]
// Directives: distribute, do, for, parallel, scope, sections, single, target,
// task, taskloop, teams
//
// [5.2:340:3-20]
// (3) The effect of the firstprivate clause is as if it is applied to one or
// more leaf constructs as follows:
//  (5) To the distribute construct if it is among the constituent constructs;
//  (6) To the teams construct if it is among the constituent constructs and the
//      distribute construct is not;
//  (8) To a worksharing construct that accepts the clause if one is among the
//      constituent constructs;
//  (9) To the taskloop construct if it is among the constituent constructs;
// (10) To the parallel construct if it is among the constituent constructs and
//      neither a taskloop construct nor a worksharing construct that accepts
//      the clause is among them;
// (12) To the target construct if it is among the constituent constructs and
//      the same list item neither appears in a lastprivate clause nor is the
//      base variable or base pointer of a list item that appears in a map
//      clause.
//
// (15) If the parallel construct is among the constituent constructs and the
// effect is not as if the firstprivate clause is applied to it by the above
// rules, then the effect is as if the shared clause with the same list item is
// applied to the parallel construct.
// (17) If the teams construct is among the constituent constructs and the
// effect is not as if the firstprivate clause is applied to it by the above
// rules, then the effect is as if the shared clause with the same list item is
// applied to the teams construct.
TEST_F(OpenMPDecompositionTest, Firstprivate1) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_firstprivate, omp::clause::Firstprivate{{x}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_parallel_sections,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "parallel shared(x)");       // (10), (15)
  ASSERT_EQ(Dir1, "sections firstprivate(x)"); // (8)
}

TEST_F(OpenMPDecompositionTest, Firstprivate2) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_firstprivate, omp::clause::Firstprivate{{x}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper,
                                  OMPD_target_teams_distribute, Clauses);
  ASSERT_EQ(Dec.output.size(), 3u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  ASSERT_EQ(Dir0, "target firstprivate(x)");     // (12)
  ASSERT_EQ(Dir1, "teams shared(x)");            // (6), (17)
  ASSERT_EQ(Dir2, "distribute firstprivate(x)"); // (5)
}

TEST_F(OpenMPDecompositionTest, Firstprivate3) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_firstprivate, omp::clause::Firstprivate{{x}}},
      {OMPC_lastprivate, omp::clause::Lastprivate{{std::nullopt, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper,
                                  OMPD_target_teams_distribute, Clauses);
  ASSERT_EQ(Dec.output.size(), 3u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  ASSERT_EQ(Dir0, "target map(2, , , , (x))"); // (12), (27)
  ASSERT_EQ(Dir1, "teams shared(x)");          // (6), (17)
  ASSERT_EQ(Dir2, "distribute firstprivate(x) lastprivate(, (x))"); // (5), (21)
}

TEST_F(OpenMPDecompositionTest, Firstprivate4) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_firstprivate, omp::clause::Firstprivate{{x}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_target_teams,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "target firstprivate(x)"); // (12)
  ASSERT_EQ(Dir1, "teams firstprivate(x)");  // (6)
}

TEST_F(OpenMPDecompositionTest, Firstprivate5) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_firstprivate, omp::clause::Firstprivate{{x}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper,
                                  OMPD_parallel_masked_taskloop, Clauses);
  ASSERT_EQ(Dec.output.size(), 3u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  ASSERT_EQ(Dir0, "parallel shared(x)"); // (10)
  ASSERT_EQ(Dir1, "masked");
  ASSERT_EQ(Dir2, "taskloop firstprivate(x)"); // (9)
}

TEST_F(OpenMPDecompositionTest, Firstprivate6) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_firstprivate, omp::clause::Firstprivate{{x}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_parallel_masked,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "parallel firstprivate(x)"); // (10)
  ASSERT_EQ(Dir1, "masked");
}

TEST_F(OpenMPDecompositionTest, Firstprivate7) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_firstprivate, omp::clause::Firstprivate{{x}}},
  };

  // Composite constructs are still decomposed.
  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_teams_distribute,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "teams shared(x)");            // (17)
  ASSERT_EQ(Dir1, "distribute firstprivate(x)"); // (5)
}

// LASTPRIVATE
// [5.2:115:7-8]
// Directives: distribute, do, for, loop, sections, simd, taskloop
//
// [5.2:340:21-30]
// (21) The effect of the lastprivate clause is as if it is applied to all leaf
// constructs that permit the clause.
// (22) If the parallel construct is among the constituent constructs and the
// list item is not also specified in the firstprivate clause, then the effect
// of the lastprivate clause is as if the shared clause with the same list item
// is applied to the parallel construct.
// (24) If the teams construct is among the constituent constructs and the list
// item is not also specified in the firstprivate clause, then the effect of the
// lastprivate clause is as if the shared clause with the same list item is
// applied to the teams construct.
// (27) If the target construct is among the constituent constructs and the list
// item is not the base variable or base pointer of a list item that appears in
// a map clause, the effect of the lastprivate clause is as if the same list
// item appears in a map clause with a map-type of tofrom.
TEST_F(OpenMPDecompositionTest, Lastprivate1) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_lastprivate, omp::clause::Lastprivate{{std::nullopt, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_parallel_sections,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "parallel shared(x)");          // (21), (22)
  ASSERT_EQ(Dir1, "sections lastprivate(, (x))"); // (21)
}

TEST_F(OpenMPDecompositionTest, Lastprivate2) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_lastprivate, omp::clause::Lastprivate{{std::nullopt, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_teams_distribute,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "teams shared(x)");               // (21), (25)
  ASSERT_EQ(Dir1, "distribute lastprivate(, (x))"); // (21)
}

TEST_F(OpenMPDecompositionTest, Lastprivate3) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_lastprivate, omp::clause::Lastprivate{{std::nullopt, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_target_parallel_do,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 3u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  ASSERT_EQ(Dir0, "target map(2, , , , (x))"); // (21), (27)
  ASSERT_EQ(Dir1, "parallel shared(x)");       // (22)
  ASSERT_EQ(Dir2, "do lastprivate(, (x))");    // (21)
}

// SHARED
// [5.2:110:5-6]
// Directives: parallel, task, taskloop, teams
//
// [5.2:340:31-32]
// (31) The effect of the shared, default, thread_limit, or order clause is as
// if it is applied to all leaf constructs that permit the clause.
TEST_F(OpenMPDecompositionTest, Shared1) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_shared, omp::clause::Shared{{x}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper,
                                  OMPD_parallel_masked_taskloop, Clauses);
  ASSERT_EQ(Dec.output.size(), 3u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  ASSERT_EQ(Dir0, "parallel shared(x)"); // (31)
  ASSERT_EQ(Dir1, "masked");             // (31)
  ASSERT_EQ(Dir2, "taskloop shared(x)"); // (31)
}

// DEFAULT
// [5.2:109:5-6]
// Directives: parallel, task, taskloop, teams
//
// [5.2:340:31-32]
// (31) The effect of the shared, default, thread_limit, or order clause is as
// if it is applied to all leaf constructs that permit the clause.
TEST_F(OpenMPDecompositionTest, Default1) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_default,
       omp::clause::Default{
           omp::clause::Default::DataSharingAttribute::Firstprivate}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper,
                                  OMPD_parallel_masked_taskloop, Clauses);
  ASSERT_EQ(Dec.output.size(), 3u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  ASSERT_EQ(Dir0, "parallel default(0)"); // (31)
  ASSERT_EQ(Dir1, "masked");              // (31)
  ASSERT_EQ(Dir2, "taskloop default(0)"); // (31)
}

// THREAD_LIMIT
// [5.2:277:14-15]
// Directives: target, teams
//
// [5.2:340:31-32]
// (31) The effect of the shared, default, thread_limit, or order clause is as
// if it is applied to all leaf constructs that permit the clause.
TEST_F(OpenMPDecompositionTest, ThreadLimit1) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_thread_limit, omp::clause::ThreadLimit{omp::ExprTy{}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper,
                                  OMPD_target_teams_distribute, Clauses);
  ASSERT_EQ(Dec.output.size(), 3u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  ASSERT_EQ(Dir0, "target thread_limit(expr)"); // (31)
  ASSERT_EQ(Dir1, "teams thread_limit(expr)");  // (31)
  ASSERT_EQ(Dir2, "distribute");                // (31)
}

// ORDER
// [5.2:234:3-4]
// Directives: distribute, do, for, loop, simd
//
// [5.2:340:31-32]
// (31) The effect of the shared, default, thread_limit, or order clause is as
// if it is applied to all leaf constructs that permit the clause.
TEST_F(OpenMPDecompositionTest, Order1) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_order,
       omp::clause::Order{{omp::clause::Order::OrderModifier::Unconstrained,
                           omp::clause::Order::Ordering::Concurrent}}},
  };

  omp::ConstructDecomposition Dec(
      AnyVersion, Helper, OMPD_target_teams_distribute_parallel_for_simd,
      Clauses);
  ASSERT_EQ(Dec.output.size(), 6u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  std::string Dir3 = stringify(Dec.output[3]);
  std::string Dir4 = stringify(Dec.output[4]);
  std::string Dir5 = stringify(Dec.output[5]);
  ASSERT_EQ(Dir0, "target"); // (31)
  ASSERT_EQ(Dir1, "teams");  // (31)
  // XXX OMP.td doesn't list "order" as allowed for "distribute"
  ASSERT_EQ(Dir2, "distribute");       // (31)
  ASSERT_EQ(Dir3, "parallel");         // (31)
  ASSERT_EQ(Dir4, "for order(1, 0)");  // (31)
  ASSERT_EQ(Dir5, "simd order(1, 0)"); // (31)
}

// ALLOCATE
// [5.2:178:7-9]
// Directives: allocators, distribute, do, for, parallel, scope, sections,
// single, target, task, taskgroup, taskloop, teams
//
// [5.2:340:33-35]
// (33) The effect of the allocate clause is as if it is applied to all leaf
// constructs that permit the clause and to which a data-sharing attribute
// clause that may create a private copy of the same list item is applied.
TEST_F(OpenMPDecompositionTest, Allocate1) {
  omp::Object x{"x"};

  // Allocate + firstprivate
  omp::List<omp::Clause> Clauses{
      {OMPC_allocate,
       omp::clause::Allocate{{std::nullopt, std::nullopt, std::nullopt, {x}}}},
      {OMPC_firstprivate, omp::clause::Firstprivate{{x}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_parallel_sections,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "parallel shared(x)");                           // (33)
  ASSERT_EQ(Dir1, "sections firstprivate(x) allocate(, , , (x))"); // (33)
}

TEST_F(OpenMPDecompositionTest, Allocate2) {
  omp::Object x{"x"};
  auto Add = red::makeOp(omp::clause::DefinedOperator::IntrinsicOperator::Add);

  // Allocate + in_reduction
  omp::List<omp::Clause> Clauses{
      {OMPC_allocate,
       omp::clause::Allocate{{std::nullopt, std::nullopt, std::nullopt, {x}}}},
      {OMPC_in_reduction, omp::clause::InReduction{{{Add}, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_target_parallel,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "target in_reduction((3), (x)) allocate(, , , (x))"); // (33)
  ASSERT_EQ(Dir1, "parallel");                                          // (33)
}

TEST_F(OpenMPDecompositionTest, Allocate3) {
  omp::Object x{"x"};

  // Allocate + linear
  omp::List<omp::Clause> Clauses{
      {OMPC_allocate,
       omp::clause::Allocate{{std::nullopt, std::nullopt, std::nullopt, {x}}}},
      {OMPC_linear,
       omp::clause::Linear{{std::nullopt, std::nullopt, std::nullopt, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_parallel_for,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  // The "shared" clause is duplicated---this isn't harmful, but it
  // should be fixed eventually.
  ASSERT_EQ(Dir0, "parallel shared(x) shared(x)"); // (33)
  ASSERT_EQ(Dir1, "for linear(, , , (x)) firstprivate(x) lastprivate(, (x)) "
                  "allocate(, , , (x))"); // (33)
}

TEST_F(OpenMPDecompositionTest, Allocate4) {
  omp::Object x{"x"};

  // Allocate + lastprivate
  omp::List<omp::Clause> Clauses{
      {OMPC_allocate,
       omp::clause::Allocate{{std::nullopt, std::nullopt, std::nullopt, {x}}}},
      {OMPC_lastprivate, omp::clause::Lastprivate{{std::nullopt, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_parallel_sections,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "parallel shared(x)");                              // (33)
  ASSERT_EQ(Dir1, "sections lastprivate(, (x)) allocate(, , , (x))"); // (33)
}

TEST_F(OpenMPDecompositionTest, Allocate5) {
  omp::Object x{"x"};

  // Allocate + private
  omp::List<omp::Clause> Clauses{
      {OMPC_allocate,
       omp::clause::Allocate{{std::nullopt, std::nullopt, std::nullopt, {x}}}},
      {OMPC_private, omp::clause::Private{{x}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_parallel_sections,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "parallel");                                // (33)
  ASSERT_EQ(Dir1, "sections private(x) allocate(, , , (x))"); // (33)
}

TEST_F(OpenMPDecompositionTest, Allocate6) {
  omp::Object x{"x"};
  auto Add = red::makeOp(omp::clause::DefinedOperator::IntrinsicOperator::Add);

  // Allocate + reduction
  omp::List<omp::Clause> Clauses{
      {OMPC_allocate,
       omp::clause::Allocate{{std::nullopt, std::nullopt, std::nullopt, {x}}}},
      {OMPC_reduction, omp::clause::Reduction{{std::nullopt, {Add}, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_parallel_sections,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "parallel shared(x)");                                 // (33)
  ASSERT_EQ(Dir1, "sections reduction(, (3), (x)) allocate(, , , (x))"); // (33)
}

// REDUCTION
// [5.2:134:17-18]
// Directives: do, for, loop, parallel, scope, sections, simd, taskloop, teams
//
// [5.2:340-341:36-13]
// (36) The effect of the reduction clause is as if it is applied to all leaf
// constructs that permit the clause, except for the following constructs:
//  (1) The parallel construct, when combined with the sections,
//      worksharing-loop, loop, or taskloop construct; and
//  (3) The teams construct, when combined with the loop construct.
// (4) For the parallel and teams constructs above, the effect of the reduction
// clause instead is as if each list item or, for any list item that is an array
// item, its corresponding base array or base pointer appears in a shared clause
// for the construct.
// (6) If the task reduction-modifier is specified, the effect is as if it only
// modifies the behavior of the reduction clause on the innermost leaf construct
// that accepts the modifier (see Section 5.5.8).
// (8) If the inscan reduction-modifier is specified, the effect is as if it
// modifies the behavior of the reduction clause on all constructs of the
// combined construct to which the clause is applied and that accept the
// modifier.
// (10) If a list item in a reduction clause on a combined target construct does
// not have the same base variable or base pointer as a list item in a map
// clause on the construct, then the effect is as if the list item in the
// reduction clause appears as a list item in a map clause with a map-type of
// tofrom.
TEST_F(OpenMPDecompositionTest, Reduction1) {
  omp::Object x{"x"};
  auto Add = red::makeOp(omp::clause::DefinedOperator::IntrinsicOperator::Add);

  omp::List<omp::Clause> Clauses{
      {OMPC_reduction, omp::clause::Reduction{{std::nullopt, {Add}, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_parallel_sections,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "parallel shared(x)");             // (36), (1), (4)
  ASSERT_EQ(Dir1, "sections reduction(, (3), (x))"); // (36)
}

TEST_F(OpenMPDecompositionTest, Reduction2) {
  omp::Object x{"x"};
  auto Add = red::makeOp(omp::clause::DefinedOperator::IntrinsicOperator::Add);

  omp::List<omp::Clause> Clauses{
      {OMPC_reduction, omp::clause::Reduction{{std::nullopt, {Add}, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_parallel_masked,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "parallel reduction(, (3), (x))"); // (36), (1), (4)
  ASSERT_EQ(Dir1, "masked");                         // (36)
}

TEST_F(OpenMPDecompositionTest, Reduction3) {
  omp::Object x{"x"};
  auto Add = red::makeOp(omp::clause::DefinedOperator::IntrinsicOperator::Add);

  omp::List<omp::Clause> Clauses{
      {OMPC_reduction, omp::clause::Reduction{{std::nullopt, {Add}, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_teams_loop, Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "teams shared(x)");            // (36), (3), (4)
  ASSERT_EQ(Dir1, "loop reduction(, (3), (x))"); // (36)
}

TEST_F(OpenMPDecompositionTest, Reduction4) {
  omp::Object x{"x"};
  auto Add = red::makeOp(omp::clause::DefinedOperator::IntrinsicOperator::Add);

  omp::List<omp::Clause> Clauses{
      {OMPC_reduction, omp::clause::Reduction{{std::nullopt, {Add}, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper,
                                  OMPD_teams_distribute_parallel_for, Clauses);
  ASSERT_EQ(Dec.output.size(), 4u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  std::string Dir3 = stringify(Dec.output[3]);
  ASSERT_EQ(Dir0, "teams reduction(, (3), (x))"); // (36), (3)
  ASSERT_EQ(Dir1, "distribute");                  // (36)
  ASSERT_EQ(Dir2, "parallel shared(x)");          // (36), (1), (4)
  ASSERT_EQ(Dir3, "for reduction(, (3), (x))");   // (36)
}

TEST_F(OpenMPDecompositionTest, Reduction5) {
  omp::Object x{"x"};
  auto Add = red::makeOp(omp::clause::DefinedOperator::IntrinsicOperator::Add);
  auto TaskMod = omp::clause::Reduction::ReductionModifier::Task;

  omp::List<omp::Clause> Clauses{
      {OMPC_reduction, omp::clause::Reduction{{TaskMod, {Add}, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper,
                                  OMPD_teams_distribute_parallel_for, Clauses);
  ASSERT_EQ(Dec.output.size(), 4u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  std::string Dir3 = stringify(Dec.output[3]);
  ASSERT_EQ(Dir0, "teams reduction(, (3), (x))"); // (36), (3), (6)
  ASSERT_EQ(Dir1, "distribute");                  // (36)
  ASSERT_EQ(Dir2, "parallel shared(x)");          // (36), (1), (4)
  ASSERT_EQ(Dir3, "for reduction(2, (3), (x))");  // (36), (6)
}

TEST_F(OpenMPDecompositionTest, Reduction6) {
  omp::Object x{"x"};
  auto Add = red::makeOp(omp::clause::DefinedOperator::IntrinsicOperator::Add);
  auto InscanMod = omp::clause::Reduction::ReductionModifier::Inscan;

  omp::List<omp::Clause> Clauses{
      {OMPC_reduction, omp::clause::Reduction{{InscanMod, {Add}, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper,
                                  OMPD_teams_distribute_parallel_for, Clauses);
  ASSERT_EQ(Dec.output.size(), 4u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  std::string Dir3 = stringify(Dec.output[3]);
  ASSERT_EQ(Dir0, "teams reduction(, (3), (x))"); // (36), (3), (8)
  ASSERT_EQ(Dir1, "distribute");                  // (36)
  ASSERT_EQ(Dir2, "parallel shared(x)");          // (36), (1), (4)
  ASSERT_EQ(Dir3, "for reduction(1, (3), (x))");  // (36), (8)
}

TEST_F(OpenMPDecompositionTest, Reduction7) {
  omp::Object x{"x"};
  auto Add = red::makeOp(omp::clause::DefinedOperator::IntrinsicOperator::Add);

  omp::List<omp::Clause> Clauses{
      {OMPC_reduction, omp::clause::Reduction{{std::nullopt, {Add}, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_target_parallel_do,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 3u);

  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  ASSERT_EQ(Dir0, "target map(2, , , , (x))"); // (36), (10)
  ASSERT_EQ(Dir1, "parallel shared(x)");       // (36), (1), (4)
  ASSERT_EQ(Dir2, "do reduction(, (3), (x))"); // (36)
}

// IF
// [5.2:72:7-9]
// Directives: cancel, parallel, simd, target, target data, target enter data,
// target exit data, target update, task, taskloop
//
// [5.2:72:15-18]
// (15) For combined or composite constructs, the if clause only applies to the
// semantics of the construct named in the directive-name-modifier.
// (16) For a combined or composite construct, if no directive-name-modifier is
// specified then the if clause applies to all constituent constructs to which
// an if clause can apply.
TEST_F(OpenMPDecompositionTest, If1) {
  omp::List<omp::Clause> Clauses{
      {OMPC_if,
       omp::clause::If{{llvm::omp::Directive::OMPD_parallel, omp::ExprTy{}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper,
                                  OMPD_target_parallel_for_simd, Clauses);
  ASSERT_EQ(Dec.output.size(), 4u);
  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  std::string Dir3 = stringify(Dec.output[3]);
  ASSERT_EQ(Dir0, "target");              // (15)
  ASSERT_EQ(Dir1, "parallel if(, expr)"); // (15)
  ASSERT_EQ(Dir2, "for");                 // (15)
  ASSERT_EQ(Dir3, "simd");                // (15)
}

TEST_F(OpenMPDecompositionTest, If2) {
  omp::List<omp::Clause> Clauses{
      {OMPC_if, omp::clause::If{{std::nullopt, omp::ExprTy{}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper,
                                  OMPD_target_parallel_for_simd, Clauses);
  ASSERT_EQ(Dec.output.size(), 4u);
  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  std::string Dir3 = stringify(Dec.output[3]);
  ASSERT_EQ(Dir0, "target if(, expr)");   // (16)
  ASSERT_EQ(Dir1, "parallel if(, expr)"); // (16)
  ASSERT_EQ(Dir2, "for");                 // (16)
  ASSERT_EQ(Dir3, "simd if(, expr)");     // (16)
}

// LINEAR
// [5.2:118:1-2]
// Directives: declare simd, do, for, simd
//
// [5.2:341:15-22]
// (15.1) The effect of the linear clause is as if it is applied to the
// innermost leaf construct.
// (15.2) Additionally, if the list item is not the iteration variable of a simd
// or worksharing-loop SIMD construct, the effect on the outer leaf constructs
// is as if the list item was specified in firstprivate and lastprivate clauses
// on the combined or composite construct, with the rules specified above
// applied.
// (19) If a list item of the linear clause is the iteration variable of a simd
// or worksharing-loop SIMD construct and it is not declared in the construct,
// the effect on the outer leaf constructs is as if the list item was specified
// in a lastprivate clause on the combined or composite construct with the rules
// specified above applied.
TEST_F(OpenMPDecompositionTest, Linear1) {
  omp::Object x{"x"};

  omp::List<omp::Clause> Clauses{
      {OMPC_linear,
       omp::clause::Linear{{std::nullopt, std::nullopt, std::nullopt, {x}}}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_for_simd, Clauses);
  ASSERT_EQ(Dec.output.size(), 2u);
  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  ASSERT_EQ(Dir0, "for firstprivate(x) lastprivate(, (x))"); // (15.1), (15.2)
  ASSERT_EQ(Dir1, "simd linear(, , , (x)) lastprivate(, (x))"); // (15.1)
}

// NOWAIT
// [5.2:308:11-13]
// Directives: dispatch, do, for, interop, scope, sections, single, target,
// target enter data, target exit data, target update, taskwait, workshare
//
// [5.2:341:23]
// (23) The effect of the nowait clause is as if it is applied to the outermost
// leaf construct that permits it.
TEST_F(OpenMPDecompositionTest, Nowait1) {
  omp::List<omp::Clause> Clauses{
      {OMPC_nowait, omp::clause::Nowait{}},
  };

  omp::ConstructDecomposition Dec(AnyVersion, Helper, OMPD_target_parallel_for,
                                  Clauses);
  ASSERT_EQ(Dec.output.size(), 3u);
  std::string Dir0 = stringify(Dec.output[0]);
  std::string Dir1 = stringify(Dec.output[1]);
  std::string Dir2 = stringify(Dec.output[2]);
  ASSERT_EQ(Dir0, "target nowait"); // (23)
  ASSERT_EQ(Dir1, "parallel");      // (23)
  ASSERT_EQ(Dir2, "for");           // (23)
}
} // namespace

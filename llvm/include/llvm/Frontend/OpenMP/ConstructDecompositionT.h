#ifndef LLVM_FRONTEND_OPENMP_COMPOUNDSPLITTERT_H
#define LLVM_FRONTEND_OPENMP_COMPOUNDSPLITTERT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Frontend/OpenMP/ClauseT.h"
#include "llvm/Frontend/OpenMP/OMP.h"

#include <iterator>
#include <list>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

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
} // namespace detail

namespace tomp {

template <typename ClauseType> struct DirectiveWithClauses {
  llvm::omp::Directive id = llvm::omp::Directive::OMPD_unknown;
  tomp::type::ListT<ClauseType> clauses;
};

template <typename ClauseType, typename HelperType>
struct ConstructDecompositionT {
  using ClauseTy = ClauseType;

  using TypeTy = typename ClauseTy::TypeTy;
  using IdTy = typename ClauseTy::IdTy;
  using ExprTy = typename ClauseTy::ExprTy;
  using HelperTy = HelperType;
  using ObjectTy = tomp::ObjectT<IdTy, ExprTy>;

  using ClauseSet = llvm::DenseSet<const ClauseTy *>;

  ConstructDecompositionT(uint32_t ver, HelperType &hlp,
                          llvm::omp::Directive dir,
                          llvm::ArrayRef<ClauseTy> clauses)
      : version(ver), construct(dir), helper(hlp) {
    for (const ClauseTy &clause : clauses)
      nodes.push_back(&clause);

    bool success = split();
    if (success) {
      // Copy the broken down directives with their clauses to the
      // output list. Copy by value, since we don't own the storage
      // with the input clauses, and the internal representation uses
      // clause addresses.
      for (auto &leaf : leafs) {
        output.push_back({leaf.id});
        auto &dwc = output.back();
        for (const ClauseTy *c : leaf.clauses)
          dwc.clauses.push_back(*c);
      }
    }
  }

  tomp::ListT<DirectiveWithClauses<ClauseType>> output;

private:
  bool split();

  struct LeafReprInternal {
    llvm::omp::Directive id = llvm::omp::Directive::OMPD_unknown;
    tomp::type::ListT<const ClauseTy *> clauses;
  };

  LeafReprInternal *findDirective(llvm::omp::Directive dirId) {
    auto found = llvm::find_if(
        leafs, [&](const LeafReprInternal &leaf) { return leaf.id == dirId; });
    return found != leafs.end() ? &*found : nullptr;
  }

  ClauseSet *findClausesWith(const ObjectTy &object) {
    if (auto found = syms.find(object.id()); found != syms.end())
      return &found->second;
    return nullptr;
  }

  template <typename S>
  ClauseTy *makeClause(llvm::omp::Clause clauseId, S &&specific) {
    implicit.push_back(ClauseTy{clauseId, std::move(specific)});
    return &implicit.back();
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

  uint32_t version;
  llvm::omp::Directive construct;
  HelperType &helper;
  ListT<LeafReprInternal> leafs;
  tomp::ListT<const ClauseTy *> nodes;
  std::list<ClauseTy> implicit; // Container for materialized implicit clauses.
                                // Inserting must preserve element addresses.
  llvm::DenseMap<IdTy, ClauseSet> syms;
  llvm::DenseSet<IdTy> mapBases;
};

// Deduction guide
template <typename ClauseType, typename HelperType>
ConstructDecompositionT(uint32_t, HelperType &, llvm::omp::Directive,
                        llvm::ArrayRef<ClauseType>)
    -> ConstructDecompositionT<ClauseType, HelperType>;

template <typename C, typename H>
void ConstructDecompositionT<C, H>::addClauseSymsToMap(const ObjectTy &object,
                                                       const ClauseTy *node) {
  syms[object.id()].insert(node);
}

template <typename C, typename H>
void ConstructDecompositionT<C, H>::addClauseSymsToMap(
    const tomp::ObjectListT<IdTy, ExprTy> &objects, const ClauseTy *node) {
  for (auto &object : objects)
    syms[object.id()].insert(node);
}

template <typename C, typename H>
void ConstructDecompositionT<C, H>::addClauseSymsToMap(const TypeTy &item,
                                                       const ClauseTy *node) {
  // Nothing to do for types.
}

template <typename C, typename H>
void ConstructDecompositionT<C, H>::addClauseSymsToMap(const ExprTy &item,
                                                       const ClauseTy *node) {
  // Nothing to do for expressions.
}

template <typename C, typename H>
void ConstructDecompositionT<C, H>::addClauseSymsToMap(
    const tomp::clause::MapT<TypeTy, IdTy, ExprTy> &item,
    const ClauseTy *node) {
  auto &objects = std::get<tomp::ObjectListT<IdTy, ExprTy>>(item.t);
  addClauseSymsToMap(objects, node);
  for (auto &object : objects) {
    if (auto base = helper.getBaseObject(object))
      mapBases.insert(base->id());
  }
}

template <typename C, typename H>
template <typename U>
void ConstructDecompositionT<C, H>::addClauseSymsToMap(
    const std::optional<U> &item, const ClauseTy *node) {
  if (item)
    addClauseSymsToMap(*item, node);
}

template <typename C, typename H>
template <typename U>
void ConstructDecompositionT<C, H>::addClauseSymsToMap(
    const tomp::ListT<U> &item, const ClauseTy *node) {
  for (auto &s : item)
    addClauseSymsToMap(s, node);
}

template <typename C, typename H>
template <typename... U, size_t... Is>
void ConstructDecompositionT<C, H>::addClauseSymsToMap(
    const std::tuple<U...> &item, const ClauseTy *node,
    std::index_sequence<Is...>) {
  (void)node; // Silence strange warning from GCC.
  (addClauseSymsToMap(std::get<Is>(item), node), ...);
}

template <typename C, typename H>
template <typename U>
std::enable_if_t<std::is_enum_v<llvm::remove_cvref_t<U>>, void>
ConstructDecompositionT<C, H>::addClauseSymsToMap(U &&item,
                                                  const ClauseTy *node) {
  // Nothing to do for enums.
}

template <typename C, typename H>
template <typename U>
std::enable_if_t<llvm::remove_cvref_t<U>::EmptyTrait::value, void>
ConstructDecompositionT<C, H>::addClauseSymsToMap(U &&item,
                                                  const ClauseTy *node) {
  // Nothing to do for an empty class.
}

template <typename C, typename H>
template <typename U>
std::enable_if_t<llvm::remove_cvref_t<U>::IncompleteTrait::value, void>
ConstructDecompositionT<C, H>::addClauseSymsToMap(U &&item,
                                                  const ClauseTy *node) {
  // Nothing to do for an incomplete class (they're empty).
}

template <typename C, typename H>
template <typename U>
std::enable_if_t<llvm::remove_cvref_t<U>::WrapperTrait::value, void>
ConstructDecompositionT<C, H>::addClauseSymsToMap(U &&item,
                                                  const ClauseTy *node) {
  addClauseSymsToMap(item.v, node);
}

template <typename C, typename H>
template <typename U>
std::enable_if_t<llvm::remove_cvref_t<U>::TupleTrait::value, void>
ConstructDecompositionT<C, H>::addClauseSymsToMap(U &&item,
                                                  const ClauseTy *node) {
  constexpr size_t tuple_size =
      std::tuple_size_v<llvm::remove_cvref_t<decltype(item.t)>>;
  addClauseSymsToMap(item.t, node, std::make_index_sequence<tuple_size>{});
}

template <typename C, typename H>
template <typename U>
std::enable_if_t<llvm::remove_cvref_t<U>::UnionTrait::value, void>
ConstructDecompositionT<C, H>::addClauseSymsToMap(U &&item,
                                                  const ClauseTy *node) {
  std::visit([&](auto &&s) { addClauseSymsToMap(s, node); }, item.u);
}

// Apply a clause to the only directive that allows it. If there are no
// directives that allow it, or if there is more that one, do not apply
// anything and return false, otherwise return true.
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyToUnique(const ClauseTy *node) {
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
template <typename C, typename H>
template <typename Iterator>
bool ConstructDecompositionT<C, H>::applyToFirst(
    const ClauseTy *node, llvm::iterator_range<Iterator> range) {
  if (range.empty())
    return false;

  for (auto &dwc : range) {
    if (!llvm::omp::isAllowedClauseForDirective(dwc.id, node->id, version))
      continue;
    dwc.clauses.push_back(node);
    return true;
  }
  return false;
}

// Apply a clause to the innermost directive that allows it. If such a
// directive does not exist, return false, otherwise return true.
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyToInnermost(const ClauseTy *node) {
  return applyToFirst(node, llvm::reverse(leafs));
}

// Apply a clause to the outermost directive that allows it. If such a
// directive does not exist, return false, otherwise return true.
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyToOutermost(const ClauseTy *node) {
  return applyToFirst(node, llvm::iterator_range(leafs));
}

template <typename C, typename H>
template <typename Predicate>
bool ConstructDecompositionT<C, H>::applyIf(const ClauseTy *node,
                                            Predicate shouldApply) {
  bool applied = false;
  for (auto &dwc : leafs) {
    if (!llvm::omp::isAllowedClauseForDirective(dwc.id, node->id, version))
      continue;
    if (!shouldApply(dwc))
      continue;
    dwc.clauses.push_back(node);
    applied = true;
  }

  return applied;
}

template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyToAll(const ClauseTy *node) {
  return applyIf(node, [](auto) { return true; });
}

template <typename C, typename H>
template <typename Clause>
bool ConstructDecompositionT<C, H>::applyClause(Clause &&clause,
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
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::CollapseT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
  // Apply COLLAPSE to the innermost directive. If it's not one that
  // allows it flag an error.
  if (!leafs.empty()) {
    auto &last = leafs.back();

    if (llvm::omp::isAllowedClauseForDirective(last.id, node->id, version)) {
      last.clauses.push_back(node);
      return true;
    }
  }

  return false;
}

// PRIVATE
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::PrivateT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
  return applyToInnermost(node);
}

// FIRSTPRIVATE
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::FirstprivateT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
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
      auto *shared = makeClause(
          llvm::omp::Clause::OMPC_shared,
          tomp::clause::SharedT<TypeTy, IdTy, ExprTy>{/*List=*/clause.v});
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
    for (auto &dwc : leafs) {
      auto found = llvm::find(worksharing, dwc.id);
      if (found != std::end(worksharing))
        return &dwc;
    }
    return static_cast<typename decltype(leafs)::value_type *>(nullptr);
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
      auto *shared = makeClause(
          llvm::omp::Clause::OMPC_shared,
          tomp::clause::SharedT<TypeTy, IdTy, ExprTy>{/*List=*/clause.v});
      hasParallel->clauses.push_back(shared);
    }
  }

  // S - To the target construct if it is among the constituent constructs
  // S   and the same list item neither appears in a lastprivate clause nor
  // S   is the base variable or base pointer of a list item that appears in
  // S   a map clause.
  auto inLastprivate = [&](const ObjectTy &object) {
    if (ClauseSet *set = findClausesWith(object)) {
      return llvm::find_if(*set, [](const ClauseTy *c) {
               return c->id == llvm::omp::Clause::OMPC_lastprivate;
             }) != set->end();
    }
    return false;
  };

  auto hasTarget = findDirective(llvm::omp::OMPD_target);
  if (hasTarget != nullptr) {
    tomp::ObjectListT<IdTy, ExprTy> objects;
    llvm::copy_if(
        clause.v, std::back_inserter(objects), [&](const ObjectTy &object) {
          return !inLastprivate(object) && !mapBases.contains(object.id());
        });
    if (!objects.empty()) {
      auto *firstp = makeClause(
          llvm::omp::Clause::OMPC_firstprivate,
          tomp::clause::FirstprivateT<TypeTy, IdTy, ExprTy>{/*List=*/objects});
      hasTarget->clauses.push_back(firstp);
      applied = true;
    }
  }

  return applied;
}

// LASTPRIVATE
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::LastprivateT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
  bool applied = false;

  // S The effect of the lastprivate clause is as if it is applied to all leaf
  // S constructs that permit the clause.
  applied = applyToAll(node);
  if (!applied)
    return false;

  auto inFirstprivate = [&](const ObjectTy &object) {
    if (ClauseSet *set = findClausesWith(object)) {
      return llvm::find_if(*set, [](const ClauseTy *c) {
               return c->id == llvm::omp::Clause::OMPC_firstprivate;
             }) != set->end();
    }
    return false;
  };

  auto &objects = std::get<tomp::ObjectListT<IdTy, ExprTy>>(clause.t);

  // Prepare list of objects that could end up in a SHARED clause.
  tomp::ObjectListT<IdTy, ExprTy> sharedObjects;
  llvm::copy_if(
      objects, std::back_inserter(sharedObjects),
      [&](const ObjectTy &object) { return !inFirstprivate(object); });

  if (!sharedObjects.empty()) {
    // S If the parallel construct is among the constituent constructs and the
    // S list item is not also specified in the firstprivate clause, then the
    // S effect of the lastprivate clause is as if the shared clause with the
    // S same list item is applied to the parallel construct.
    if (auto hasParallel = findDirective(llvm::omp::OMPD_parallel)) {
      auto *shared = makeClause(
          llvm::omp::Clause::OMPC_shared,
          tomp::clause::SharedT<TypeTy, IdTy, ExprTy>{/*List=*/sharedObjects});
      hasParallel->clauses.push_back(shared);
      applied = true;
    }

    // S If the teams construct is among the constituent constructs and the
    // S list item is not also specified in the firstprivate clause, then the
    // S effect of the lastprivate clause is as if the shared clause with the
    // S same list item is applied to the teams construct.
    if (auto hasTeams = findDirective(llvm::omp::OMPD_teams)) {
      auto *shared = makeClause(
          llvm::omp::Clause::OMPC_shared,
          tomp::clause::SharedT<TypeTy, IdTy, ExprTy>{/*List=*/sharedObjects});
      hasTeams->clauses.push_back(shared);
      applied = true;
    }
  }

  // S If the target construct is among the constituent constructs and the
  // S list item is not the base variable or base pointer of a list item that
  // S appears in a map clause, the effect of the lastprivate clause is as if
  // S the same list item appears in a map clause with a map-type of tofrom.
  if (auto hasTarget = findDirective(llvm::omp::OMPD_target)) {
    tomp::ObjectListT<IdTy, ExprTy> tofrom;
    llvm::copy_if(objects, std::back_inserter(tofrom),
                  [&](const ObjectTy &object) {
                    return !mapBases.contains(object.id());
                  });

    if (!tofrom.empty()) {
      using MapType =
          typename tomp::clause::MapT<TypeTy, IdTy, ExprTy>::MapType;
      auto *map =
          makeClause(llvm::omp::Clause::OMPC_map,
                     tomp::clause::MapT<TypeTy, IdTy, ExprTy>{
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
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::SharedT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
  // Apply SHARED to the all leafs that allow it.
  return applyToAll(node);
}

// DEFAULT
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::DefaultT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
  // Apply DEFAULT to the all leafs that allow it.
  return applyToAll(node);
}

// THREAD_LIMIT
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::ThreadLimitT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
  // Apply THREAD_LIMIT to the all leafs that allow it.
  return applyToAll(node);
}

// ORDER
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::OrderT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
  // Apply ORDER to the all leafs that allow it.
  return applyToAll(node);
}

// ALLOCATE
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::AllocateT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
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

  bool applied = applyIf(node, [&](const auto &dwc) {
    return llvm::any_of(dwc.clauses, [&](const ClauseTy *n) {
      return canMakePrivateCopy(n->id);
    });
  });

  return applied;
}

// REDUCTION
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::ReductionT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
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

  auto &objects = std::get<tomp::ObjectListT<IdTy, ExprTy>>(clause.t);

  tomp::ObjectListT<IdTy, ExprTy> sharedObjects;
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
      auto *shared = makeClause(
          llvm::omp::Clause::OMPC_shared,
          tomp::clause::SharedT<TypeTy, IdTy, ExprTy>{/*List=*/sharedObjects});
      hasParallel->clauses.push_back(shared);
    }
    if (hasTeams && !applyToTeams) {
      auto *shared = makeClause(
          llvm::omp::Clause::OMPC_shared,
          tomp::clause::SharedT<TypeTy, IdTy, ExprTy>{/*List=*/sharedObjects});
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

  bool applied = applyIf(node, [&](auto &dwc) {
    if (!applyToParallel && &dwc == hasParallel)
      return false;
    if (!applyToTeams && &dwc == hasTeams)
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
    tomp::ObjectListT<IdTy, ExprTy> tofrom;
    llvm::copy_if(objects, std::back_inserter(tofrom),
                  [&](const ObjectTy &object) {
                    if (auto maybeBase = helper.getBaseObject(object))
                      return !mapBases.contains(maybeBase->id());
                    return !mapBases.contains(object.id()); // XXX is this ok?
                  });
    if (!tofrom.empty()) {
      using MapType =
          typename tomp::clause::MapT<TypeTy, IdTy, ExprTy>::MapType;
      auto *map = makeClause(
          llvm::omp::Clause::OMPC_map,
          tomp::clause::MapT<TypeTy, IdTy, ExprTy>{
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
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::IfT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
  using DirectiveNameModifier =
      typename clause::IfT<TypeTy, IdTy, ExprTy>::DirectiveNameModifier;
  auto &modifier = std::get<std::optional<DirectiveNameModifier>>(clause.t);

  if (modifier) {
    llvm::omp::Directive dirId = *modifier;

    if (auto *hasDir = findDirective(dirId)) {
      hasDir->clauses.push_back(node);
      return true;
    }
    return false;
  }

  return applyToAll(node);
}

// LINEAR
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::LinearT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
  // S The effect of the linear clause is as if it is applied to the innermost
  // S leaf construct.
  if (!applyToInnermost(node))
    return false;

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

  std::optional<ObjectTy> iterVar = helper.getLoopIterVar();
  const auto &objects = std::get<tomp::ObjectListT<IdTy, ExprTy>>(clause.t);

  // Lists of objects that will be used to construct FIRSTPRIVATE and
  // LASTPRIVATE clauses.
  tomp::ObjectListT<IdTy, ExprTy> first, last;

  for (const ObjectTy &object : objects) {
    last.push_back(object);
    if (iterVar && object.id() != iterVar->id())
      first.push_back(object);
  }

  if (!first.empty()) {
    auto *firstp = makeClause(
        llvm::omp::Clause::OMPC_firstprivate,
        tomp::clause::FirstprivateT<TypeTy, IdTy, ExprTy>{/*List=*/first});
    nodes.push_back(firstp); // Appending to the main clause list.
  }
  if (!last.empty()) {
    auto *lastp =
        makeClause(llvm::omp::Clause::OMPC_lastprivate,
                   tomp::clause::LastprivateT<TypeTy, IdTy, ExprTy>{
                       {/*LastprivateModifier=*/std::nullopt, /*List=*/last}});
    nodes.push_back(lastp); // Appending to the main clause list.
  }
  return true;
}

// NOWAIT
template <typename C, typename H>
bool ConstructDecompositionT<C, H>::applyClause(
    const tomp::clause::NowaitT<TypeTy, IdTy, ExprTy> &clause,
    const ClauseTy *node) {
  return applyToOutermost(node);
}

template <typename C, typename H> bool ConstructDecompositionT<C, H>::split() {
  bool success = true;

  for (llvm::omp::Directive leaf :
       llvm::omp::getLeafConstructsOrSelf(construct))
    leafs.push_back(LeafReprInternal{leaf, /*clauses=*/{}});

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
    success = success &&
              applyClause(std::get<tomp::clause::LinearT<TypeTy, IdTy, ExprTy>>(
                              node->u),
                          node);
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

#endif // LLVM_FRONTEND_OPENMP_COMPOUNDSPLITTERT_H

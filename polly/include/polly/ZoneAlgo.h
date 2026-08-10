//===------ ZoneAlgo.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Derive information about array elements between statements ("Zones").
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_ZONEALGO_H
#define POLLY_ZONEALGO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "isl/isl-noexceptions.h"
#include <memory>

namespace llvm {
class Value;
class LoopInfo;
class Loop;
class PHINode;
class raw_ostream;
} // namespace llvm

namespace polly {
class Scop;
class ScopStmt;
class MemoryAccess;
class ScopArrayInfo;

/// Return only the mappings that map to known values.
///
/// @param UMap { [] -> ValInst[] }
///
/// @return { [] -> ValInst[] }
isl::union_map filterKnownValInst(const isl::union_map &UMap);

/// Base class for algorithms based on zones, like DeLICM.
class ZoneAlgorithm {
protected:
  /// The name of the pass this is used from. Used for optimization remarks.
  const char *PassName;

  /// Hold a reference to the isl_ctx to avoid it being freed before we released
  /// all of the isl objects.
  ///
  /// This must be declared before any other member that holds an isl object.
  /// This guarantees that the shared_ptr and its isl_ctx is destructed last,
  /// after all other members free'd the isl objects they were holding.
  std::shared_ptr<isl_ctx> IslCtx;

  /// Cached reaching definitions for each ScopStmt.
  ///
  /// Use getScalarReachingDefinition() to get its contents.
  llvm::DenseMap<ScopStmt *, isl::map> ScalarReachDefZone;

  /// The analyzed Scop.
  Scop *S;

  /// LoopInfo analysis used to determine whether values are synthesizable.
  llvm::LoopInfo *LI;

  /// Parameter space that does not need realignment.
  isl::space ParamSpace;

  /// Space the schedule maps to.
  isl::space ScatterSpace;

  /// Cached version of the schedule and domains.
  isl::union_map Schedule;

  /// Combined access relations of all MemoryKind::Array READ accesses.
  /// { DomainRead[] -> Element[] }
  isl::union_map AllReads;

  /// The loaded values (llvm::LoadInst) of all reads.
  /// { [Element[] -> DomainRead[]] -> ValInst[] }
  isl::union_map AllReadValInst;

  /// Combined access relations of all MemoryKind::Array, MAY_WRITE accesses.
  /// { DomainMayWrite[] -> Element[] }
  isl::union_map AllMayWrites;

  /// Combined access relations of all MemoryKind::Array, MUST_WRITE accesses.
  /// { DomainMustWrite[] -> Element[] }
  isl::union_map AllMustWrites;

  /// Combined access relations of all MK_Array write accesses (union of
  /// AllMayWrites and AllMustWrites).
  /// { DomainWrite[] -> Element[] }
  isl::union_map AllWrites;

  /// The value instances written to array elements of all write accesses.
  /// { [Element[] -> DomainWrite[]] -> ValInst[] }
  isl::union_map AllWriteValInst;

  /// All reaching definitions for  MemoryKind::Array writes.
  /// { [Element[] -> Zone[]] -> DomainWrite[] }
  isl::union_map WriteReachDefZone;

  /// Map llvm::Values to an isl identifier.
  /// Used with -polly-use-llvm-names=false as an alternative method to get
  /// unique ids that do not depend on pointer values.
  llvm::DenseMap<llvm::Value *, isl::id> ValueIds;

  /// Set of array elements that can be reliably used for zone analysis.
  /// { Element[] }
  isl::union_set CompatibleElts;

  /// List of PHIs that may transitively refer to themselves.
  ///
  /// Computing them would require a polyhedral transitive closure operation,
  /// for which isl may only return an approximation. For correctness, we always
  /// require an exact result. Hence, we exclude such PHIs.
  llvm::SmallPtrSet<llvm::PHINode *, 4> RecursivePHIs;

  /// PHIs that have been computed.
  ///
  /// Computed PHIs are replaced by their incoming values using #NormalizeMap.
  llvm::DenseSet<llvm::PHINode *> ComputedPHIs;

  /// For computed PHIs, contains the ValInst they stand for.
  ///
  /// To show an example, assume the following PHINode:
  ///
  ///   Stmt:
  ///     %phi = phi double [%val1, %bb1], [%val2, %bb2]
  ///
  /// It's ValInst is:
  ///
  ///   { [Stmt[i] -> phi[]] }
  ///
  /// The value %phi will be either %val1 or %val2, depending on whether in
  /// iteration i %bb1 or %bb2 has been executed before. In SCoPs, this can be
  /// determined at compile-time, and the result stored in #NormalizeMap. For
  /// the previous example, it could be:
  ///
  ///   { [Stmt[i] -> phi[]] -> [Stmt[0] -> val1[]];
  ///     [Stmt[i] -> phi[]] -> [Stmt[i] -> val2[]] : i > 0 }
  ///
  /// Only ValInsts in #ComputedPHIs are present in this map. Other values are
  /// assumed to represent themselves. This is to avoid adding lots of identity
  /// entries to this map.
  ///
  /// { PHIValInst[] -> IncomingValInst[] }
  isl::union_map NormalizeMap;

  /// Cache for computePerPHI(const ScopArrayInfo *)
  llvm::SmallDenseMap<llvm::PHINode *, isl::union_map> PerPHIMaps;

  /// A cache for getDefToTarget().
  llvm::DenseMap<std::pair<ScopStmt *, ScopStmt *>, isl::map> DefToTargetCache;

  /// Prepare the object before computing the zones of @p S.
  ///
  /// @param PassName Name of the pass using this analysis.
  /// @param S        The SCoP to process.
  /// @param LI       LoopInfo analysis used to determine synthesizable values.
  ZoneAlgorithm(const char *PassName, Scop *S, llvm::LoopInfo *LI);

private:
  /// Find the array elements that violate the zone analysis assumptions.
  ///
  /// What violates our assumptions:
  /// - A load after a write of the same location; we assume that all reads
  ///   occur before the writes.
  /// - Two writes to the same location; we cannot model the order in which
  ///   these occur.
  ///
  /// Scalar reads implicitly always occur before other accesses therefore never
  /// violate the first condition. There is also at most one write to a scalar,
  /// satisfying the second condition.
  ///
  /// @param Stmt                  The statement to be analyzed.
  /// @param[out] IncompatibleElts Receives the elements that are not
  ///                              zone-analysis compatible.
  /// @param[out]                  AllElts receives all encountered elements.
  void collectIncompatibleElts(ScopStmt *Stmt, isl::union_set &IncompatibleElts,
                               isl::union_set &AllElts);

  void addArrayReadAccess(MemoryAccess *MA);

  /// Return the ValInst write by a (must-)write access. Returns the 'unknown'
  /// ValInst if there is no single ValInst[] the array element written to will
  /// have.
  ///
  /// @return { ValInst[] }
  isl::union_map getWrittenValue(MemoryAccess *MA, isl::map AccRel);

  void addArrayWriteAccess(MemoryAccess *MA);

  /// For an llvm::Value defined in @p DefStmt, compute the RAW dependency for a
  /// use in every instance of @p UseStmt.
  ///
  /// @param UseStmt Statement a scalar is used in.
  /// @param DefStmt Statement a scalar is defined in.
  ///
  /// @return { DomainUse[] -> DomainDef[] }
  isl::map computeUseToDefFlowDependency(ScopStmt *UseStmt, ScopStmt *DefStmt);

protected:
  isl::union_set makeEmptyUnionSet() const;

  isl::union_map makeEmptyUnionMap() const;

  /// For each 'execution' of a PHINode, get the incoming block that was
  /// executed before.
  ///
  /// For each PHI instance we can directly determine which was the incoming
  /// block, and hence derive which value the PHI has.
  ///
  /// @param SAI The ScopArrayInfo representing the PHI's storage.
  ///
  /// @return { DomainPHIRead[] -> DomainPHIWrite[] }
  isl::union_map computePerPHI(const polly::ScopArrayInfo *SAI);

  /// Find the array elements that can be used for zone analysis.
  void collectCompatibleElts();

  /// Get the schedule for @p Stmt.
  ///
  /// The domain of the result is as narrow as possible.
  isl::map getScatterFor(ScopStmt *Stmt) const;

  /// Get the schedule of @p MA's parent statement.
  isl::map getScatterFor(MemoryAccess *MA) const;

  /// Get the schedule for the statement instances of @p Domain.
  isl::union_map getScatterFor(isl::union_set Domain) const;

  /// Get the schedule for the statement instances of @p Domain.
  isl::map getScatterFor(isl::set Domain) const;

  /// Get the domain of @p Stmt.
  isl::set getDomainFor(ScopStmt *Stmt) const;

  /// Get the domain @p MA's parent statement.
  isl::set getDomainFor(MemoryAccess *MA) const;

  /// Get the access relation of @p MA.
  ///
  /// The domain of the result is as narrow as possible.
  isl::map getAccessRelationFor(MemoryAccess *MA) const;

  /// Get a domain translation map from a (scalar) definition to the statement
  /// where the definition is being moved to.
  ///
  /// @p TargetStmt can also be seen at an llvm::Use of an llvm::Value in
  /// @p DefStmt. In addition, we allow transitive uses:
  ///
  /// DefStmt -> MiddleStmt -> TargetStmt
  ///
  /// where an operand tree of instructions in DefStmt and MiddleStmt are to be
  /// moved to TargetStmt. To be generally correct, we also need to know all the
  /// intermediate statements. However, we make use of the fact that
  /// ForwardOpTree currently does not support a move from a loop body across
  /// its header such that only the first definition and the target statement
  /// are relevant.
  ///
  /// @param DefStmt    Statement from where a definition might be moved from.
  /// @param TargetStmt Statement where the definition is potentially being
  ///                   moved to (should contain a use of that definition).
  ///
  /// @return { DomainDef[] -> DomainTarget[] }
  isl::map getDefToTarget(ScopStmt *DefStmt, ScopStmt *TargetStmt);

  /// Get the reaching definition of a scalar defined in @p Stmt.
  ///
  /// Note that this does not depend on the llvm::Instruction, only on the
  /// statement it is defined in. Therefore the same computation can be reused.
  ///
  /// @param Stmt The statement in which a scalar is defined.
  ///
  /// @return { Scatter[] -> DomainDef[] }
  isl::map getScalarReachingDefinition(ScopStmt *Stmt);

  /// Get the reaching definition of a scalar defined in @p DefDomain.
  ///
  /// @param DomainDef { DomainDef[] }
  ///              The write statements to get the reaching definition for.
  ///
  /// @return { Scatter[] -> DomainDef[] }
  isl::map getScalarReachingDefinition(isl::set DomainDef);

  /// Create a statement-to-unknown value mapping.
  ///
  /// @param Stmt The statement whose instances are mapped to unknown.
  ///
  /// @return { Domain[] -> ValInst[] }
  isl::map makeUnknownForDomain(ScopStmt *Stmt) const;

  /// Create an isl_id that represents @p V.
  isl::id makeValueId(llvm::Value *V);

  /// Create the space for an llvm::Value that is available everywhere.
  isl::space makeValueSpace(llvm::Value *V);

  /// Create a set with the llvm::Value @p V which is available everywhere.
  isl::set makeValueSet(llvm::Value *V);

  /// Create a mapping from a statement instance to the instance of an
  /// llvm::Value that can be used in there.
  ///
  /// Although LLVM IR uses single static assignment, llvm::Values can have
  /// different contents in loops, when they get redefined in the last
  /// iteration. This function tries to get the statement instance of the
  /// previous definition, relative to a user.
  ///
  /// Example:
  /// for (int i = 0; i < N; i += 1) {
  /// DEF:
  ///    int v = A[i];
  /// USE:
  ///    use(v);
  ///  }
  ///
  /// The value instance used by statement instance USE[i] is DEF[i]. Hence,
  /// makeValInst returns:
  ///
  /// { USE[i] -> [DEF[i] -> v[]] : 0 <= i < N }
  ///
  /// @param Val       The value to get the instance of.
  /// @param UserStmt  The statement that uses @p Val. Can be nullptr.
  /// @param Scope     Loop the using instruction resides in.
  /// @param IsCertain Pass true if the definition of @p Val is a
  ///                  MUST_WRITE or false if the write is conditional.
  ///
  /// @return { DomainUse[] -> ValInst[] }
  isl::map makeValInst(llvm::Value *Val, ScopStmt *UserStmt, llvm::Loop *Scope,
                       bool IsCertain = true);

  /// Create and normalize a ValInst.
  ///
  /// @see makeValInst
  /// @see normalizeValInst
  /// @see #NormalizedPHI
  isl::union_map makeNormalizedValInst(llvm::Value *Val, ScopStmt *UserStmt,
                                       llvm::Loop *Scope,
                                       bool IsCertain = true);

  /// Return whether @p MA can be used for transformations (e.g. OpTree load
  /// forwarding, DeLICM mapping).
  bool isCompatibleAccess(MemoryAccess *MA);

  /// Compute the different zones.
  void computeCommon();

  ///  Compute the normalization map that replaces PHIs by their incoming
  ///  values.
  ///
  /// @see #NormalizeMap
  void computeNormalizedPHIs();

  /// Print the current state of all MemoryAccesses to @p.
  void printAccesses(llvm::raw_ostream &OS, int Indent = 0) const;

  /// Is @p MA a PHI READ access that can be normalized?
  ///
  /// @see #NormalizeMap
  bool isNormalizable(MemoryAccess *MA);

  /// @{
  /// Determine whether the argument does not map to any computed PHI. Those
  /// should have been replaced by their incoming values.
  ///
  /// @see #NormalizedPHI
  isl::boolean isNormalized(isl::map Map);
  isl::boolean isNormalized(isl::union_map Map);
  /// @}

public:
  /// Return the SCoP this object is analyzing.
  Scop *getScop() const { return S; }

  /// A reaching definition zone is known to have the definition's written value
  /// if the definition is a MUST_WRITE.
  ///
  /// @return { [Element[] -> Zone[]] -> ValInst[] }
  isl::union_map computeKnownFromMustWrites() const;

  /// A reaching definition zone is known to be the same value as any load that
  /// reads from that array element in that period.
  ///
  /// @return { [Element[] -> Zone[]] -> ValInst[] }
  isl::union_map computeKnownFromLoad() const;

  /// Compute which value an array element stores at every instant.
  ///
  /// @param FromWrite Use stores as source of information.
  /// @param FromRead  Use loads as source of information.
  ///
  /// @return { [Element[] -> Zone[]] -> ValInst[] }
  isl::union_map computeKnown(bool FromWrite, bool FromRead) const;
};

/// Create a domain-to-unknown value mapping.
///
/// Value instances that do not represent a specific value are represented by an
/// unnamed tuple of 0 dimensions. Its meaning depends on the context. It can
/// either mean a specific but unknown value which cannot be represented by
/// other means. It conflicts with itself because those two unknown ValInsts may
/// have different concrete values at runtime.
///
/// The other meaning is an arbitrary or wildcard value that can be chosen
/// freely, like LLVM's undef. If matched with an unknown ValInst, there is no
/// conflict.
///
/// @param Domain { Domain[] }
///
/// @return { Domain[] -> ValInst[] }
isl::union_map makeUnknownForDomain(isl::union_set Domain);
} // namespace polly

#endif /* POLLY_ZONEALGO_H */

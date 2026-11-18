//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the BTF type deduplication algorithm, a port of the algorithm
// from libbpf (btf_dedup.c, BSD-licensed).
//
// The algorithm runs in 5 passes:
//   1. String deduplication
//   2. Primitive and composite type dedup and forward declaration resolution
//   3. Reference type dedup (PTR, TYPEDEF, VOLATILE, etc.)
//   4. Type compaction (remove dups, assign sequential IDs)
//   5. Type ID remapping (fix all references to use new IDs)
//
// For struct/union types, a DFS-based type graph equivalence check is used
// with a "hypothetical map" to handle recursive/cyclic types.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/BTF/BTFDedup.h"
#include "BTFTypeUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/DebugInfo/BTF/BTF.h"
#include "llvm/DebugInfo/BTF/BTFBuilder.h"
#include <functional>

#define DEBUG_TYPE "btf-dedup"

using namespace llvm;

namespace {

// Sentinel value: type has not been processed yet.
constexpr uint32_t BTF_UNPROCESSED = UINT32_MAX;
constexpr uint32_t BTF_IN_PROGRESS = UINT32_MAX - 1;

/// State for the BTF deduplication algorithm.
class BTFDedupState {
  BTFBuilder &Builder;

  // Equivalence map: Map[i] = canonical type ID for type i.
  // Unprocessed types do not yet have a canonical representative.
  std::vector<uint32_t> Map;

  // Hypothetical map for recursive type comparison.
  // HypotMap[canonical graph ID] = matching ID in the candidate graph.
  // Reset between each top-level comparison.
  std::vector<uint32_t> HypotMap;

  // List of types currently in the hypothetical map (for fast reset).
  SmallVector<uint32_t, 0> HypotList;

  // String dedup: maps string content to canonical offset.
  StringMap<uint32_t> StringDedup;

  // New string offsets: NewStrOff[old_offset] = new_offset after dedup.
  DenseMap<uint32_t, uint32_t> NewStrOff;

  // Buckets: hash -> list of canonical type IDs with that hash.
  DenseMap<uint64_t, SmallVector<uint32_t, 4>> HashBuckets;

  // After compaction: OldToNew[old_id] = new_id.
  std::vector<uint32_t> OldToNew;

  static bool isPrimitiveKind(uint32_t Kind) {
    switch (Kind) {
    case BTF::BTF_KIND_INT:
    case BTF::BTF_KIND_FLOAT:
    case BTF::BTF_KIND_ENUM:
    case BTF::BTF_KIND_ENUM64:
    case BTF::BTF_KIND_FWD:
      return true;
    default:
      return false;
    }
  }

  // Returns true if this is a composite kind that needs DFS comparison.
  static bool isCompositeKind(uint32_t Kind) {
    return Kind == BTF::BTF_KIND_STRUCT || Kind == BTF::BTF_KIND_UNION;
  }

  static bool isEnumKind(uint32_t Kind) {
    return Kind == BTF::BTF_KIND_ENUM || Kind == BTF::BTF_KIND_ENUM64;
  }

  static bool isEnumFwd(const BTF::CommonType &T) {
    return isEnumKind(T.getKind()) && T.getVlen() == 0;
  }

  static uint32_t fwdKind(const BTF::CommonType &T) {
    return T.Info & BTF::FWD_UNION_FLAG ? BTF::BTF_KIND_UNION
                                        : BTF::BTF_KIND_STRUCT;
  }

  bool isMapped(uint32_t Id) const { return Map[Id] < BTF_IN_PROGRESS; }

  // Get the canonical representative for a type ID.
  uint32_t resolve(uint32_t Id) const {
    while (isMapped(Id) && Map[Id] != Id)
      Id = Map[Id];
    return Id;
  }

  // Hash reference types only after their referenced IDs are canonical.
  uint64_t hashType(uint32_t Id);

  // Hash helpers for specific kinds.
  uint64_t hashCommon(const BTF::CommonType *T);
  uint64_t hashStruct(const BTF::CommonType *T);
  uint64_t hashFuncProto(const BTF::CommonType *T);
  uint64_t hashArray(const BTF::CommonType *T);

  // Compare two graphs using the hypothetical map for cycle handling.
  bool isEquiv(uint32_t CandId, uint32_t CanonId);
  bool isIdentical(uint32_t LHS, uint32_t RHS, unsigned Depth);
  void mergeHypot();

  // Compare record fields, delegating type references to the caller.
  using RefEqual = function_ref<bool(uint32_t, uint32_t)>;
  bool compareTypes(uint32_t CandId, uint32_t CanonId, RefEqual EqualRef,
                    bool MatchEnumFwds = false);
  bool isEquivCommon(const BTF::CommonType *Cand, const BTF::CommonType *Canon);
  bool isEquivStruct(const BTF::CommonType *Cand, const BTF::CommonType *Canon,
                     RefEqual EqualRef);
  bool isEquivEnum(const BTF::CommonType *Cand, const BTF::CommonType *Canon);
  bool isEquivEnum64(const BTF::CommonType *Cand, const BTF::CommonType *Canon);
  bool isEquivFuncProto(const BTF::CommonType *Cand,
                        const BTF::CommonType *Canon, RefEqual EqualRef);
  bool isEquivArray(const BTF::CommonType *Cand, const BTF::CommonType *Canon,
                    RefEqual EqualRef);

  ArrayRef<uint8_t> tail(uint32_t Id) const {
    return Builder.getTypeBytes(Id).drop_front(sizeof(BTF::CommonType));
  }

  // Reset the hypothetical map.
  void clearHypot() {
    for (uint32_t Id : HypotList)
      HypotMap[Id] = BTF_UNPROCESSED;
    HypotList.clear();
  }

  uint32_t dedupStrOff(uint32_t Offset) const {
    auto It = NewStrOff.find(Offset);
    return It != NewStrOff.end() ? It->second : Offset;
  }

  // The five passes.
  Error dedupStrings();
  void dedupPrimitives();
  void dedupComposites();
  void resolveFwds();
  Error dedupRef(uint32_t Id);
  Error dedupRefs();
  Error compact();

public:
  BTFDedupState(BTFBuilder &B) : Builder(B) {}
  Error run();
};

//===----------------------------------------------------------------------===//
// Hashing
//===----------------------------------------------------------------------===//

uint64_t BTFDedupState::hashCommon(const BTF::CommonType *T) {
  return hash_combine(T->Info, dedupStrOff(T->NameOff), T->Size);
}

uint64_t BTFDedupState::hashStruct(const BTF::CommonType *T) {
  // Hash name + size + member names (NOT member types — those are checked
  // during equivalence comparison).
  uint64_t H = hashCommon(T);
  auto *Members = reinterpret_cast<const BTF::BTFMember *>(
      reinterpret_cast<const uint8_t *>(T) + sizeof(BTF::CommonType));
  for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
    H = hash_combine(H, dedupStrOff(Members[I].NameOff), Members[I].Offset);
  return H;
}

uint64_t BTFDedupState::hashFuncProto(const BTF::CommonType *T) {
  uint64_t H = hash_combine(T->Info, dedupStrOff(T->NameOff), resolve(T->Type));
  auto *Params = reinterpret_cast<const BTF::BTFParam *>(
      reinterpret_cast<const uint8_t *>(T) + sizeof(BTF::CommonType));
  for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
    H = hash_combine(H, dedupStrOff(Params[I].NameOff),
                     resolve(Params[I].Type));
  return H;
}

uint64_t BTFDedupState::hashArray(const BTF::CommonType *T) {
  auto *Arr = reinterpret_cast<const BTF::BTFArray *>(
      reinterpret_cast<const uint8_t *>(T) + sizeof(BTF::CommonType));
  return hash_combine(hashCommon(T), Arr->Nelems, resolve(Arr->ElemType),
                      resolve(Arr->IndexType));
}

uint64_t BTFDedupState::hashType(uint32_t Id) {
  const BTF::CommonType *T = Builder.findType(Id);
  if (!T)
    return 0;

  switch (T->getKind()) {
  case BTF::BTF_KIND_INT:
    return hash_combine(hashCommon(T), hash_value(tail(Id)));
  case BTF::BTF_KIND_FLOAT:
  case BTF::BTF_KIND_FWD:
    return hashCommon(T);
  case BTF::BTF_KIND_ENUM:
  case BTF::BTF_KIND_ENUM64:
    // A declaration can match either enum kind, regardless of size or flags.
    return hash_combine(dedupStrOff(T->NameOff));
  case BTF::BTF_KIND_STRUCT:
  case BTF::BTF_KIND_UNION:
    return hashStruct(T);
  case BTF::BTF_KIND_TYPEDEF:
    return hash_combine(T->Info, dedupStrOff(T->NameOff));
  case BTF::BTF_KIND_FUNC_PROTO:
    return hashFuncProto(T);
  case BTF::BTF_KIND_ARRAY:
    return hashArray(T);
  case BTF::BTF_KIND_DECL_TAG:
    return hash_combine(T->Info, dedupStrOff(T->NameOff), resolve(T->Type),
                        hash_value(tail(Id)));
  default:
    return hash_combine(T->Info, dedupStrOff(T->NameOff), resolve(T->Type));
  }
}

//===----------------------------------------------------------------------===//
// Equivalence checking
//===----------------------------------------------------------------------===//

bool BTFDedupState::isEquivCommon(const BTF::CommonType *Cand,
                                  const BTF::CommonType *Canon) {
  return Cand->Info == Canon->Info &&
         dedupStrOff(Cand->NameOff) == dedupStrOff(Canon->NameOff) &&
         Cand->Size == Canon->Size;
}

bool BTFDedupState::isEquivEnum(const BTF::CommonType *Cand,
                                const BTF::CommonType *Canon) {
  if (!isEquivCommon(Cand, Canon))
    return false;

  auto *CandVals = reinterpret_cast<const BTF::BTFEnum *>(
      reinterpret_cast<const uint8_t *>(Cand) + sizeof(BTF::CommonType));
  auto *CanonVals = reinterpret_cast<const BTF::BTFEnum *>(
      reinterpret_cast<const uint8_t *>(Canon) + sizeof(BTF::CommonType));

  for (unsigned I = 0, N = Cand->getVlen(); I < N; ++I) {
    if (dedupStrOff(CandVals[I].NameOff) != dedupStrOff(CanonVals[I].NameOff) ||
        CandVals[I].Val != CanonVals[I].Val)
      return false;
  }
  return true;
}

bool BTFDedupState::isEquivEnum64(const BTF::CommonType *Cand,
                                  const BTF::CommonType *Canon) {
  if (!isEquivCommon(Cand, Canon))
    return false;

  auto *CandVals = reinterpret_cast<const BTF::BTFEnum64 *>(
      reinterpret_cast<const uint8_t *>(Cand) + sizeof(BTF::CommonType));
  auto *CanonVals = reinterpret_cast<const BTF::BTFEnum64 *>(
      reinterpret_cast<const uint8_t *>(Canon) + sizeof(BTF::CommonType));

  for (unsigned I = 0, N = Cand->getVlen(); I < N; ++I) {
    if (dedupStrOff(CandVals[I].NameOff) != dedupStrOff(CanonVals[I].NameOff) ||
        CandVals[I].Val_Lo32 != CanonVals[I].Val_Lo32 ||
        CandVals[I].Val_Hi32 != CanonVals[I].Val_Hi32)
      return false;
  }
  return true;
}

bool BTFDedupState::isEquivStruct(const BTF::CommonType *Cand,
                                  const BTF::CommonType *Canon,
                                  RefEqual EqualRef) {
  if (Cand->Size != Canon->Size)
    return false;

  auto *CandMembers = reinterpret_cast<const BTF::BTFMember *>(
      reinterpret_cast<const uint8_t *>(Cand) + sizeof(BTF::CommonType));
  auto *CanonMembers = reinterpret_cast<const BTF::BTFMember *>(
      reinterpret_cast<const uint8_t *>(Canon) + sizeof(BTF::CommonType));

  for (unsigned I = 0, N = Cand->getVlen(); I < N; ++I) {
    if (dedupStrOff(CandMembers[I].NameOff) !=
            dedupStrOff(CanonMembers[I].NameOff) ||
        CandMembers[I].Offset != CanonMembers[I].Offset)
      return false;
    if (!EqualRef(CandMembers[I].Type, CanonMembers[I].Type))
      return false;
  }
  return true;
}

bool BTFDedupState::isEquivFuncProto(const BTF::CommonType *Cand,
                                     const BTF::CommonType *Canon,
                                     RefEqual EqualRef) {
  if (!EqualRef(Cand->Type, Canon->Type))
    return false;

  auto *CandParams = reinterpret_cast<const BTF::BTFParam *>(
      reinterpret_cast<const uint8_t *>(Cand) + sizeof(BTF::CommonType));
  auto *CanonParams = reinterpret_cast<const BTF::BTFParam *>(
      reinterpret_cast<const uint8_t *>(Canon) + sizeof(BTF::CommonType));

  for (unsigned I = 0, N = Cand->getVlen(); I < N; ++I) {
    if (dedupStrOff(CandParams[I].NameOff) !=
        dedupStrOff(CanonParams[I].NameOff))
      return false;
    if (!EqualRef(CandParams[I].Type, CanonParams[I].Type))
      return false;
  }
  return true;
}

bool BTFDedupState::isEquivArray(const BTF::CommonType *Cand,
                                 const BTF::CommonType *Canon,
                                 RefEqual EqualRef) {
  if (Cand->Size != Canon->Size)
    return false;

  auto *CandArr = reinterpret_cast<const BTF::BTFArray *>(
      reinterpret_cast<const uint8_t *>(Cand) + sizeof(BTF::CommonType));
  auto *CanonArr = reinterpret_cast<const BTF::BTFArray *>(
      reinterpret_cast<const uint8_t *>(Canon) + sizeof(BTF::CommonType));

  if (CandArr->Nelems != CanonArr->Nelems)
    return false;
  return EqualRef(CandArr->IndexType, CanonArr->IndexType) &&
         EqualRef(CandArr->ElemType, CanonArr->ElemType);
}

bool BTFDedupState::compareTypes(uint32_t CandId, uint32_t CanonId,
                                 RefEqual EqualRef, bool MatchEnumFwds) {
  const BTF::CommonType *Cand = Builder.findType(CandId);
  const BTF::CommonType *Canon = Builder.findType(CanonId);
  if (!Cand || !Canon || Cand->getKind() != Canon->getKind() ||
      dedupStrOff(Cand->NameOff) != dedupStrOff(Canon->NameOff))
    return false;
  if (MatchEnumFwds && (isEnumFwd(*Cand) || isEnumFwd(*Canon)))
    return true;
  if (Cand->Info != Canon->Info)
    return false;

  switch (Cand->getKind()) {
  case BTF::BTF_KIND_INT:
    return Cand->Size == Canon->Size && tail(CandId) == tail(CanonId);
  case BTF::BTF_KIND_FLOAT:
  case BTF::BTF_KIND_FWD:
    return Cand->Size == Canon->Size;

  case BTF::BTF_KIND_ENUM:
    return isEquivEnum(Cand, Canon);

  case BTF::BTF_KIND_ENUM64:
    return isEquivEnum64(Cand, Canon);

  case BTF::BTF_KIND_STRUCT:
  case BTF::BTF_KIND_UNION:
    return isEquivStruct(Cand, Canon, EqualRef);

  case BTF::BTF_KIND_FUNC_PROTO:
    return isEquivFuncProto(Cand, Canon, EqualRef);

  case BTF::BTF_KIND_ARRAY:
    return isEquivArray(Cand, Canon, EqualRef);

  case BTF::BTF_KIND_PTR:
  case BTF::BTF_KIND_TYPEDEF:
  case BTF::BTF_KIND_VOLATILE:
  case BTF::BTF_KIND_CONST:
  case BTF::BTF_KIND_RESTRICT:
  case BTF::BTF_KIND_TYPE_TAG:
  case BTF::BTF_KIND_FUNC:
    return EqualRef(Cand->Type, Canon->Type);

  case BTF::BTF_KIND_DECL_TAG:
    return tail(CandId) == tail(CanonId) && EqualRef(Cand->Type, Canon->Type);

  default:
    return false;
  }
}

bool BTFDedupState::isIdentical(uint32_t LHS, uint32_t RHS, unsigned Depth) {
  if (LHS == RHS)
    return true;
  return Depth && compareTypes(
                      LHS, RHS,
                      [&](uint32_t A, uint32_t B) {
                        return isIdentical(A, B, Depth - 1);
                      },
                      /*MatchEnumFwds=*/true);
}

bool BTFDedupState::isEquiv(uint32_t CandId, uint32_t CanonId) {
  uint32_t ResolvedCanon = resolve(CanonId);
  if (resolve(CandId) == ResolvedCanon)
    return true;
  if (!CandId || !CanonId)
    return false;

  // Keep the shape of the canonical graph, except where a FWD has already
  // gained a definition. Known canonical IDs provide the fast path above.
  if (Builder.findType(CanonId)->getKind() == BTF::BTF_KIND_FWD &&
      Builder.findType(ResolvedCanon)->getKind() != BTF::BTF_KIND_FWD)
    CanonId = ResolvedCanon;

  // Different IDs can represent identical records within one input graph.
  if (HypotMap[CanonId] != BTF_UNPROCESSED)
    return isIdentical(HypotMap[CanonId], CandId, 16);
  HypotMap[CanonId] = CandId;
  HypotList.push_back(CanonId);

  const BTF::CommonType *Cand = Builder.findType(CandId);
  const BTF::CommonType *Canon = Builder.findType(CanonId);
  if (Cand->getKind() != Canon->getKind() &&
      dedupStrOff(Cand->NameOff) == dedupStrOff(Canon->NameOff)) {
    if (Cand->getKind() == BTF::BTF_KIND_FWD)
      return fwdKind(*Cand) == Canon->getKind();
    if (Canon->getKind() == BTF::BTF_KIND_FWD)
      return fwdKind(*Canon) == Cand->getKind();
  }
  if (Cand->getKind() == BTF::BTF_KIND_DECL_TAG)
    return false;
  return compareTypes(
      CandId, CanonId, [this](uint32_t A, uint32_t B) { return isEquiv(A, B); },
      /*MatchEnumFwds=*/true);
}

void BTFDedupState::mergeHypot() {
  for (uint32_t Id : HypotList) {
    uint32_t Canon = resolve(Id);
    uint32_t Cand = resolve(HypotMap[Id]);
    if (Canon == Cand)
      continue;
    uint32_t CanonKind = Builder.findType(Canon)->getKind();
    uint32_t CandKind = Builder.findType(Cand)->getKind();
    if (CanonKind == BTF::BTF_KIND_FWD && isCompositeKind(CandKind))
      Map[Canon] = Cand;
    else if (CandKind == BTF::BTF_KIND_FWD && isCompositeKind(CanonKind))
      Map[Cand] = Canon;
    else if (isCompositeKind(CandKind) && isCompositeKind(CanonKind) &&
             isMapped(Canon) && !isMapped(Cand))
      Map[Cand] = Canon;
  }
}

//===----------------------------------------------------------------------===//
// Pass 1: String deduplication
//===----------------------------------------------------------------------===//

Error BTFDedupState::dedupStrings() {
  StringDedup[""] = 0;

  for (uint32_t Id = 1; Id <= Builder.typesCount(); ++Id) {
    const BTF::CommonType *T = Builder.findType(Id);
    if (!T)
      continue;

    auto DedupStr = [&](uint32_t Off) {
      if (NewStrOff.count(Off))
        return;
      StringRef S = Builder.findString(Off);
      auto It = StringDedup.find(S);
      if (It != StringDedup.end()) {
        NewStrOff[Off] = It->second;
      } else {
        StringDedup[S] = Off;
        NewStrOff[Off] = Off;
      }
    };

    DedupStr(T->NameOff);

    const uint8_t *TailPtr =
        reinterpret_cast<const uint8_t *>(T) + sizeof(BTF::CommonType);
    switch (T->getKind()) {
    case BTF::BTF_KIND_STRUCT:
    case BTF::BTF_KIND_UNION: {
      auto *M = reinterpret_cast<const BTF::BTFMember *>(TailPtr);
      for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
        DedupStr(M[I].NameOff);
      break;
    }
    case BTF::BTF_KIND_ENUM: {
      auto *E = reinterpret_cast<const BTF::BTFEnum *>(TailPtr);
      for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
        DedupStr(E[I].NameOff);
      break;
    }
    case BTF::BTF_KIND_ENUM64: {
      auto *E = reinterpret_cast<const BTF::BTFEnum64 *>(TailPtr);
      for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
        DedupStr(E[I].NameOff);
      break;
    }
    case BTF::BTF_KIND_FUNC_PROTO: {
      auto *P = reinterpret_cast<const BTF::BTFParam *>(TailPtr);
      for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
        DedupStr(P[I].NameOff);
      break;
    }
    default:
      break;
    }
  }

  return Error::success();
}

//===----------------------------------------------------------------------===//
// Pass 2: Primitive and composite type dedup
//===----------------------------------------------------------------------===//

void BTFDedupState::dedupPrimitives() {
  uint32_t N = Builder.typesCount();
  for (uint32_t Id = 1; Id <= N; ++Id) {
    const BTF::CommonType *T = Builder.findType(Id);
    uint32_t Kind = T->getKind();
    // VAR and DATASEC represent distinct objects, not interchangeable types.
    if (Kind == BTF::BTF_KIND_VAR || Kind == BTF::BTF_KIND_DATASEC) {
      Map[Id] = Id;
      continue;
    }
    if (!isPrimitiveKind(Kind))
      continue;

    auto &Bucket = HashBuckets[hashType(Id)];
    Map[Id] = Id;
    for (uint32_t CanonId : llvm::reverse(Bucket)) {
      if (compareTypes(Id, CanonId, std::equal_to<uint32_t>())) {
        Map[Id] = CanonId;
        break;
      }
      const BTF::CommonType *Canon = Builder.findType(CanonId);
      if (!isEnumKind(Kind) || !isEnumKind(Canon->getKind()) ||
          dedupStrOff(T->NameOff) != dedupStrOff(Canon->NameOff))
        continue;
      if (isEnumFwd(*T)) {
        Map[Id] = CanonId;
        break;
      }
      if (isEnumFwd(*Canon))
        Map[CanonId] = Id;
    }
    if (Map[Id] == Id)
      Bucket.push_back(Id);
  }
}

void BTFDedupState::dedupComposites() {
  uint32_t N = Builder.typesCount();
  for (uint32_t Id = 1; Id <= N; ++Id) {
    if (isMapped(Id))
      continue;
    uint32_t Kind = Builder.findType(Id)->getKind();
    // Typedefs can participate in matching a forward declaration to its
    // definition, so compare their graphs before resolving standalone FWDs.
    if (!isCompositeKind(Kind) && Kind != BTF::BTF_KIND_TYPEDEF)
      continue;
    auto &Bucket = HashBuckets[hashType(Id)];
    for (uint32_t CanonId : llvm::reverse(Bucket)) {
      if (Builder.findType(CanonId)->getKind() != Kind)
        continue;
      clearHypot();
      if (isEquiv(Id, CanonId)) {
        mergeHypot();
        Map[Id] = CanonId;
        break;
      }
    }
    if (!isMapped(Id)) {
      Map[Id] = Id;
      Bucket.push_back(Id);
    }
  }
}

void BTFDedupState::resolveFwds() {
  DenseMap<uint32_t, uint32_t> Definitions;
  uint32_t N = Builder.typesCount();
  for (uint32_t Id = 1; Id <= N; ++Id) {
    const BTF::CommonType *T = Builder.findType(Id);
    if (Map[Id] != Id || !isCompositeKind(T->getKind()))
      continue;
    // Zero means the name is ambiguous, even across struct and union kinds.
    auto [It, Inserted] = Definitions.try_emplace(dedupStrOff(T->NameOff), Id);
    if (!Inserted)
      It->second = 0;
  }
  for (uint32_t Id = 1; Id <= N; ++Id) {
    const BTF::CommonType *T = Builder.findType(Id);
    if (Map[Id] != Id || T->getKind() != BTF::BTF_KIND_FWD)
      continue;
    uint32_t Def = Definitions.lookup(dedupStrOff(T->NameOff));
    if (Def && Builder.findType(Def)->getKind() == fwdKind(*T))
      Map[Id] = Def;
  }
}

//===----------------------------------------------------------------------===//
// Pass 3: Reference type dedup
//===----------------------------------------------------------------------===//

Error BTFDedupState::dedupRef(uint32_t Id) {
  if (isMapped(Id))
    return Error::success();
  if (Map[Id] == BTF_IN_PROGRESS)
    return createStringError("cycle in BTF reference types (type ID %u)", Id);
  Map[Id] = BTF_IN_PROGRESS;

  const BTF::CommonType *T = Builder.findType(Id);
  uint32_t Kind = T->getKind();
  if (BTFBuilder::hasTypeRef(Kind))
    if (Error E = dedupRef(T->Type))
      return E;
  if (Kind == BTF::BTF_KIND_ARRAY) {
    auto *A = reinterpret_cast<const BTF::BTFArray *>(tail(Id).data());
    if (Error E = dedupRef(A->ElemType))
      return E;
    if (Error E = dedupRef(A->IndexType))
      return E;
  } else if (Kind == BTF::BTF_KIND_FUNC_PROTO) {
    auto *P = reinterpret_cast<const BTF::BTFParam *>(tail(Id).data());
    for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
      if (Error E = dedupRef(P[I].Type))
        return E;
  }

  // Canonicalize dependencies before hashing, even for forward type IDs.
  auto &Bucket = HashBuckets[hashType(Id)];
  for (uint32_t CanonId : Bucket) {
    if (compareTypes(Id, CanonId, [this](uint32_t A, uint32_t B) {
          return resolve(A) == resolve(B);
        })) {
      Map[Id] = CanonId;
      return Error::success();
    }
  }
  Map[Id] = Id;
  Bucket.push_back(Id);
  return Error::success();
}

Error BTFDedupState::dedupRefs() {
  for (uint32_t Id = 1, N = Builder.typesCount(); Id <= N; ++Id)
    if (Error E = dedupRef(Id))
      return E;
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Pass 4-5: Compaction and remapping
//===----------------------------------------------------------------------===//

Error BTFDedupState::compact() {
  uint32_t N = Builder.typesCount();
  BTFBuilder NewBuilder;
  DenseMap<uint32_t, uint32_t> StrMap;
  auto MapStr = [&](uint32_t OldOff) -> uint32_t {
    uint32_t DedupOff = dedupStrOff(OldOff);
    auto It = StrMap.find(DedupOff);
    if (It != StrMap.end())
      return It->second;
    StringRef S = Builder.findString(DedupOff);
    uint32_t NewOff = NewBuilder.addString(S);
    StrMap[DedupOff] = NewOff;
    return NewOff;
  };

  StrMap[0] = 0;
  OldToNew.assign(N + 1, 0);
  for (uint32_t Id = 1; Id <= N; ++Id) {
    if (Map[Id] != Id)
      continue;

    const BTF::CommonType *T = Builder.findType(Id);
    if (!T)
      continue;

    BTF::CommonType NewHeader = *T;
    NewHeader.NameOff = MapStr(T->NameOff);
    uint32_t NewId = NewBuilder.addType(NewHeader);
    OldToNew[Id] = NewId;

    ArrayRef<uint8_t> TypeBytes = Builder.getTypeBytes(Id);
    if (TypeBytes.size() > sizeof(BTF::CommonType)) {
      ArrayRef<uint8_t> Tail = TypeBytes.slice(sizeof(BTF::CommonType));
      SmallVector<uint8_t, 64> TailCopy(Tail.begin(), Tail.end());
      uint8_t *TailPtr = TailCopy.data();

      switch (T->getKind()) {
      case BTF::BTF_KIND_STRUCT:
      case BTF::BTF_KIND_UNION: {
        auto *M = reinterpret_cast<BTF::BTFMember *>(TailPtr);
        for (unsigned I = 0, VN = T->getVlen(); I < VN; ++I)
          M[I].NameOff = MapStr(M[I].NameOff);
        break;
      }
      case BTF::BTF_KIND_ENUM: {
        auto *E = reinterpret_cast<BTF::BTFEnum *>(TailPtr);
        for (unsigned I = 0, VN = T->getVlen(); I < VN; ++I)
          E[I].NameOff = MapStr(E[I].NameOff);
        break;
      }
      case BTF::BTF_KIND_ENUM64: {
        auto *E = reinterpret_cast<BTF::BTFEnum64 *>(TailPtr);
        for (unsigned I = 0, VN = T->getVlen(); I < VN; ++I)
          E[I].NameOff = MapStr(E[I].NameOff);
        break;
      }
      case BTF::BTF_KIND_FUNC_PROTO: {
        auto *P = reinterpret_cast<BTF::BTFParam *>(TailPtr);
        for (unsigned I = 0, VN = T->getVlen(); I < VN; ++I)
          P[I].NameOff = MapStr(P[I].NameOff);
        break;
      }
      default:
        break;
      }

      for (uint8_t B : TailCopy)
        NewBuilder.addTail(B);
    }
  }

  for (uint32_t Id = 1; Id <= N; ++Id) {
    if (OldToNew[Id] != 0)
      continue;
    uint32_t CanonId = resolve(Id);
    OldToNew[Id] = OldToNew[CanonId];
  }

  for (uint32_t NewId = 1; NewId <= NewBuilder.typesCount(); ++NewId) {
    MutableArrayRef<uint8_t> Bytes = NewBuilder.getMutableTypeBytes(NewId);
    if (Bytes.empty())
      continue;

    auto *T = reinterpret_cast<BTF::CommonType *>(Bytes.data());
    uint8_t *TailPtr = Bytes.data() + sizeof(BTF::CommonType);

    if (BTFBuilder::hasTypeRef(T->getKind()) && T->Type != 0) {
      if (T->Type < OldToNew.size())
        T->Type = OldToNew[T->Type];
    }

    switch (T->getKind()) {
    case BTF::BTF_KIND_ARRAY: {
      auto *A = reinterpret_cast<BTF::BTFArray *>(TailPtr);
      if (A->ElemType != 0 && A->ElemType < OldToNew.size())
        A->ElemType = OldToNew[A->ElemType];
      if (A->IndexType != 0 && A->IndexType < OldToNew.size())
        A->IndexType = OldToNew[A->IndexType];
      break;
    }
    case BTF::BTF_KIND_STRUCT:
    case BTF::BTF_KIND_UNION: {
      auto *M = reinterpret_cast<BTF::BTFMember *>(TailPtr);
      for (unsigned I = 0, VN = T->getVlen(); I < VN; ++I)
        if (M[I].Type != 0 && M[I].Type < OldToNew.size())
          M[I].Type = OldToNew[M[I].Type];
      break;
    }
    case BTF::BTF_KIND_FUNC_PROTO: {
      auto *P = reinterpret_cast<BTF::BTFParam *>(TailPtr);
      for (unsigned I = 0, VN = T->getVlen(); I < VN; ++I)
        if (P[I].Type != 0 && P[I].Type < OldToNew.size())
          P[I].Type = OldToNew[P[I].Type];
      break;
    }
    case BTF::BTF_KIND_DATASEC: {
      auto *D = reinterpret_cast<BTF::BTFDataSec *>(TailPtr);
      for (unsigned I = 0, VN = T->getVlen(); I < VN; ++I)
        if (D[I].Type != 0 && D[I].Type < OldToNew.size())
          D[I].Type = OldToNew[D[I].Type];
      break;
    }
    default:
      break;
    }
  }

  Builder = std::move(NewBuilder);
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Main entry point
//===----------------------------------------------------------------------===//

Error BTFDedupState::run() {
  uint32_t N = Builder.typesCount();
  if (N == 0)
    return Error::success();

  // Reject invalid references before equivalence checking indexes HypotMap,
  // and before any pass mutates the builder.
  for (uint32_t Id = 1; Id <= N; ++Id)
    if (Error E = BTF::validateTypeRefs(*Builder.findType(Id), N))
      return E;

  Map.assign(N + 1, BTF_UNPROCESSED);
  Map[0] = 0;
  HypotMap.assign(N + 1, BTF_UNPROCESSED);

  if (Error E = dedupStrings())
    return E;

  dedupPrimitives();
  dedupComposites();
  resolveFwds();
  if (Error E = dedupRefs())
    return E;
  if (Error E = compact())
    return E;

  return Error::success();
}

} // anonymous namespace

Error llvm::BTF::dedup(BTFBuilder &Builder) {
  BTFDedupState State(Builder);
  return State.run();
}

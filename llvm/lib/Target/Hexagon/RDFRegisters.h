//===--- RDFRegisters.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_RDFREGISTERS_H
#define LLVM_LIB_TARGET_HEXAGON_RDFREGISTERS_H

#include "llvm/ADT/BitVector.h"
#include "llvm/Target/TargetRegisterInfo.h"

#include <set>
#include <unordered_map>
#include <vector>

namespace llvm {
namespace rdf {

  typedef uint32_t RegisterId;

  // Template class for a map translating uint32_t into arbitrary types.
  // The map will act like an indexed set: upon insertion of a new object,
  // it will automatically assign a new index to it. Index of 0 is treated
  // as invalid and is never allocated.
  template <typename T, unsigned N = 32>
  struct IndexedSet {
    IndexedSet() : Map() { Map.reserve(N); }

    T get(uint32_t Idx) const {
      // Index Idx corresponds to Map[Idx-1].
      assert(Idx != 0 && !Map.empty() && Idx-1 < Map.size());
      return Map[Idx-1];
    }

    uint32_t insert(T Val) {
      // Linear search.
      auto F = llvm::find(Map, Val);
      if (F != Map.end())
        return F - Map.begin() + 1;
      Map.push_back(Val);
      return Map.size();  // Return actual_index + 1.
    }

    uint32_t find(T Val) const {
      auto F = llvm::find(Map, Val);
      assert(F != Map.end());
      return F - Map.begin() + 1;
    }

  private:
    std::vector<T> Map;
  };

  struct RegisterRef {
    RegisterId Reg = 0;
    LaneBitmask Mask = LaneBitmask::getNone();

    RegisterRef() = default;
    explicit RegisterRef(RegisterId R, LaneBitmask M = LaneBitmask::getAll())
      : Reg(R), Mask(R != 0 ? M : LaneBitmask::getNone()) {}

    operator bool() const {
      return Reg != 0 && Mask.any();
    }
    bool operator== (const RegisterRef &RR) const {
      return Reg == RR.Reg && Mask == RR.Mask;
    }
    bool operator!= (const RegisterRef &RR) const {
      return !operator==(RR);
    }
    bool operator< (const RegisterRef &RR) const {
      return Reg < RR.Reg || (Reg == RR.Reg && Mask < RR.Mask);
    }
  };


  struct PhysicalRegisterInfo {
    PhysicalRegisterInfo(const TargetRegisterInfo &tri,
                         const MachineFunction &mf);

    static bool isRegMaskId(RegisterId R) {
      return TargetRegisterInfo::isStackSlot(R);
    }
    RegisterId getRegMaskId(const uint32_t *RM) const {
      return TargetRegisterInfo::index2StackSlot(RegMasks.find(RM));
    }
    const uint32_t *getRegMaskBits(RegisterId R) const {
      return RegMasks.get(TargetRegisterInfo::stackSlot2Index(R));
    }
    RegisterRef normalize(RegisterRef RR) const;

    bool alias(RegisterRef RA, RegisterRef RB) const {
      if (!isRegMaskId(RA.Reg))
        return !isRegMaskId(RB.Reg) ? aliasRR(RA, RB) : aliasRM(RA, RB);
      return !isRegMaskId(RB.Reg) ? aliasRM(RB, RA) : aliasMM(RA, RB);
    }
    std::set<RegisterId> getAliasSet(RegisterId Reg) const;

    const TargetRegisterInfo &getTRI() const { return TRI; }

  private:
    struct RegInfo {
      unsigned MaxSuper = 0;
      const TargetRegisterClass *RegClass = nullptr;
    };

    const TargetRegisterInfo &TRI;
    std::vector<RegInfo> RegInfos;
    IndexedSet<const uint32_t*> RegMasks;

    bool aliasRR(RegisterRef RA, RegisterRef RB) const;
    bool aliasRM(RegisterRef RR, RegisterRef RM) const;
    bool aliasMM(RegisterRef RM, RegisterRef RN) const;
  };


  struct RegisterAggr {
    RegisterAggr(const PhysicalRegisterInfo &pri)
        : ExpAliasUnits(pri.getTRI().getNumRegUnits()), PRI(pri) {}
    RegisterAggr(const RegisterAggr &RG) = default;

    bool empty() const { return Masks.empty(); }
    bool hasAliasOf(RegisterRef RR) const;
    bool hasCoverOf(RegisterRef RR) const;
    static bool isCoverOf(RegisterRef RA, RegisterRef RB,
                          const PhysicalRegisterInfo &PRI) {
      return RegisterAggr(PRI).insert(RA).hasCoverOf(RB);
    }

    RegisterAggr &insert(RegisterRef RR);
    RegisterAggr &insert(const RegisterAggr &RG);
    RegisterAggr &clear(RegisterRef RR);
    RegisterAggr &clear(const RegisterAggr &RG);

    RegisterRef clearIn(RegisterRef RR) const;

    void print(raw_ostream &OS) const;

  private:
    typedef std::unordered_map<RegisterId, LaneBitmask> MapType;

  public:
    typedef MapType::const_iterator iterator;
    iterator begin() const { return Masks.begin(); }
    iterator end() const { return Masks.end(); }

  private:
    MapType Masks;
    BitVector ExpAliasUnits; // Register units for explicit aliases.
    bool CheckUnits = false;
    const PhysicalRegisterInfo &PRI;
  };


  // Optionally print the lane mask, if it is not ~0.
  struct PrintLaneMaskOpt {
    PrintLaneMaskOpt(LaneBitmask M) : Mask(M) {}
    LaneBitmask Mask;
  };
  raw_ostream &operator<< (raw_ostream &OS, const PrintLaneMaskOpt &P);

} // namespace rdf
} // namespace llvm

#endif


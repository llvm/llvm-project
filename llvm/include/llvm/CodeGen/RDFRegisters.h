//===- RDFRegisters.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_RDFREGISTERS_H
#define LLVM_LIB_TARGET_HEXAGON_RDFREGISTERS_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/LaneBitmask.h"
#include <cassert>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace llvm {

class MachineFunction;
class raw_ostream;

namespace rdf {

  using RegisterId = uint32_t;

  // Template class for a map translating uint32_t into arbitrary types.
  // The map will act like an indexed set: upon insertion of a new object,
  // it will automatically assign a new index to it. Index of 0 is treated
  // as invalid and is never allocated.
  template <typename T, unsigned N = 32>
  struct IndexedSet {
    IndexedSet() { Map.reserve(N); }

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

    uint32_t size() const { return Map.size(); }

    using const_iterator = typename std::vector<T>::const_iterator;

    const_iterator begin() const { return Map.begin(); }
    const_iterator end() const { return Map.end(); }

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

    size_t hash() const {
      return std::hash<RegisterId>{}(Reg) ^
             std::hash<LaneBitmask::Type>{}(Mask.getAsInteger());
    }
  };


  struct PhysicalRegisterInfo {
    PhysicalRegisterInfo(const TargetRegisterInfo &tri,
                         const MachineFunction &mf);

    static bool isRegMaskId(RegisterId R) {
      return Register::isStackSlot(R);
    }

    RegisterId getRegMaskId(const uint32_t *RM) const {
      return Register::index2StackSlot(RegMasks.find(RM));
    }

    const uint32_t *getRegMaskBits(RegisterId R) const {
      return RegMasks.get(Register::stackSlot2Index(R));
    }

    LLVM_ATTRIBUTE_DEPRECATED(RegisterRef normalize(RegisterRef RR),
      "This function is now an identity function");
    RegisterRef normalize(RegisterRef RR) const {
      return RR;
    }

    bool alias(RegisterRef RA, RegisterRef RB) const {
      if (!isRegMaskId(RA.Reg))
        return !isRegMaskId(RB.Reg) ? aliasRR(RA, RB) : aliasRM(RA, RB);
      return !isRegMaskId(RB.Reg) ? aliasRM(RB, RA) : aliasMM(RA, RB);
    }

    std::set<RegisterId> getAliasSet(RegisterId Reg) const;

    RegisterRef getRefForUnit(uint32_t U) const {
      return RegisterRef(UnitInfos[U].Reg, UnitInfos[U].Mask);
    }

    const BitVector &getMaskUnits(RegisterId MaskId) const {
      return MaskInfos[Register::stackSlot2Index(MaskId)].Units;
    }

    const BitVector &getUnitAliases(uint32_t U) const {
      return AliasInfos[U].Regs;
    }

    RegisterRef mapTo(RegisterRef RR, unsigned R) const;
    const TargetRegisterInfo &getTRI() const { return TRI; }

  private:
    struct RegInfo {
      const TargetRegisterClass *RegClass = nullptr;
    };
    struct UnitInfo {
      RegisterId Reg = 0;
      LaneBitmask Mask;
    };
    struct MaskInfo {
      BitVector Units;
    };
    struct AliasInfo {
      BitVector Regs;
    };

    const TargetRegisterInfo &TRI;
    IndexedSet<const uint32_t*> RegMasks;
    std::vector<RegInfo> RegInfos;
    std::vector<UnitInfo> UnitInfos;
    std::vector<MaskInfo> MaskInfos;
    std::vector<AliasInfo> AliasInfos;

    bool aliasRR(RegisterRef RA, RegisterRef RB) const;
    bool aliasRM(RegisterRef RR, RegisterRef RM) const;
    bool aliasMM(RegisterRef RM, RegisterRef RN) const;
  };

  struct RegisterAggr {
    RegisterAggr(const PhysicalRegisterInfo &pri)
        : Units(pri.getTRI().getNumRegUnits()), PRI(pri) {}
    RegisterAggr(const RegisterAggr &RG) = default;

    unsigned count() const { return Units.count(); }
    bool empty() const { return Units.none(); }
    bool hasAliasOf(RegisterRef RR) const;
    bool hasCoverOf(RegisterRef RR) const;

    bool operator==(const RegisterAggr &A) const {
      return DenseMapInfo<BitVector>::isEqual(Units, A.Units);
    }

    static bool isCoverOf(RegisterRef RA, RegisterRef RB,
                          const PhysicalRegisterInfo &PRI) {
      return RegisterAggr(PRI).insert(RA).hasCoverOf(RB);
    }

    RegisterAggr &insert(RegisterRef RR);
    RegisterAggr &insert(const RegisterAggr &RG);
    RegisterAggr &intersect(RegisterRef RR);
    RegisterAggr &intersect(const RegisterAggr &RG);
    RegisterAggr &clear(RegisterRef RR);
    RegisterAggr &clear(const RegisterAggr &RG);

    RegisterRef intersectWith(RegisterRef RR) const;
    RegisterRef clearIn(RegisterRef RR) const;
    RegisterRef makeRegRef() const;

    size_t hash() const {
      return DenseMapInfo<BitVector>::getHashValue(Units);
    }

    void print(raw_ostream &OS) const;

    struct rr_iterator {
      using MapType = std::map<RegisterId, LaneBitmask>;

    private:
      MapType Masks;
      MapType::iterator Pos;
      unsigned Index;
      const RegisterAggr *Owner;

    public:
      rr_iterator(const RegisterAggr &RG, bool End);

      RegisterRef operator*() const {
        return RegisterRef(Pos->first, Pos->second);
      }

      rr_iterator &operator++() {
        ++Pos;
        ++Index;
        return *this;
      }

      bool operator==(const rr_iterator &I) const {
        assert(Owner == I.Owner);
        (void)Owner;
        return Index == I.Index;
      }

      bool operator!=(const rr_iterator &I) const {
        return !(*this == I);
      }
    };

    rr_iterator rr_begin() const {
      return rr_iterator(*this, false);
    }
    rr_iterator rr_end() const {
      return rr_iterator(*this, true);
    }

  private:
    BitVector Units;
    const PhysicalRegisterInfo &PRI;
  };

  // Optionally print the lane mask, if it is not ~0.
  struct PrintLaneMaskOpt {
    PrintLaneMaskOpt(LaneBitmask M) : Mask(M) {}
    LaneBitmask Mask;
  };
  raw_ostream &operator<< (raw_ostream &OS, const PrintLaneMaskOpt &P);

  raw_ostream &operator<< (raw_ostream &OS, const RegisterAggr &A);
} // end namespace rdf

} // end namespace llvm

namespace std {
  template <> struct hash<llvm::rdf::RegisterRef> {
    size_t operator()(llvm::rdf::RegisterRef A) const {
      return A.hash();
    }
  };
  template <> struct hash<llvm::rdf::RegisterAggr> {
    size_t operator()(const llvm::rdf::RegisterAggr &A) const {
      return A.hash();
    }
  };
  template <> struct equal_to<llvm::rdf::RegisterAggr> {
    bool operator()(const llvm::rdf::RegisterAggr &A,
                    const llvm::rdf::RegisterAggr &B) const {
      return A == B;
    }
  };
}
#endif // LLVM_LIB_TARGET_HEXAGON_RDFREGISTERS_H

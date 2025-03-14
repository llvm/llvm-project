#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/MC/LaneBitmask.h"

#include "llvm/CodeGen/ScheduleDAG.h" // For SUnit.

namespace llvm {
class MachineFunction;
class LiveIntervals;
class MachineRegisterInfo;
class SIRegisterInfo;
class SIInstrInfo;
class MachineInstr;
class MachineBasicBlock;
template <typename GraphType> class GraphWriter;
class SUnit;
class IntEqClasses;
class Twine;

using LiveSet = llvm::DenseMap<unsigned, llvm::LaneBitmask>;

// SubExp and BlockExpDag.
struct SubExp {
  // Keep original order for sunits.
  std::vector<llvm::MachineInstr *> SUnits;
  llvm::DenseSet<unsigned> TopRegs;
  llvm::DenseSet<llvm::MachineInstr *> BottomRoots;
  llvm::DenseSet<unsigned> BottomRegs;
  bool IsMultiDefOutput = false;
  bool IsHasTerminatorInst = false;
  bool IsUseIncomingReg = false;
  bool IsMoveIntoLoop = false;
  bool IsNotSafeToCopy = false;
  bool IsHasMemInst = false;
  bool IsHoist = false;
  // If temp/out reg is used by inst not in the subExp, cannot move since not
  // all users will be move. But OK to clone.
  bool IsCloneOnly = false;
  bool IsTouchSCC = false;
  llvm::MachineBasicBlock *FromBB;
  llvm::MachineBasicBlock *ToBB;
  unsigned SInputSize;
  unsigned VInputSize;
  unsigned SOutputSize;
  unsigned VOutputSize;
  unsigned SMaxSize;
  unsigned VMaxSize;
  LiveSet InputLive;
  LiveSet OutputLive;
  bool isSafeToMove(const llvm::MachineRegisterInfo &MRI, bool IsMoveUp) const;
  void calcMaxPressure(const llvm::MachineRegisterInfo &MRI,
                       const llvm::SIRegisterInfo *SIRI);
  void dump(const llvm::MachineRegisterInfo &MRI,
            const llvm::SIRegisterInfo *SIRI) const;
  bool modifiesRegister(unsigned Reg, const llvm::SIRegisterInfo *SIRI) const;
};

struct ExpDag {
  ExpDag(const llvm::MachineRegisterInfo &MRI, const llvm::SIRegisterInfo *SIRI,
         const llvm::SIInstrInfo *SIII, const bool IsJoinInput);
  const llvm::MachineRegisterInfo &MRI;
  const llvm::SIRegisterInfo *SIRI;
  const llvm::SIInstrInfo *SIII;
  const bool IsJoinInputToSubExp;

  std::vector<llvm::SUnit> SUnits; ///< The scheduling units.
  llvm::DenseMap<llvm::MachineInstr *, llvm::SUnit *> MISUnitMap;
  llvm::DenseMap<llvm::SUnit *, llvm::MachineInstr *> SUnitMIMap;
  llvm::DenseMap<unsigned, llvm::SUnit *> InputSUnitMap;
  llvm::DenseMap<llvm::SUnit *, unsigned> SUnitInputMap;
  std::vector<SubExp> SubExps;
  template <typename T>
  void build(const LiveSet &InputLiveReg, const LiveSet &OutputLiveReg,
             T &Insts);
  void dump();
  void viewGraph(const llvm::Twine &Name, const llvm::Twine &Title) const;
  /// Returns a label for an SUnit node in a visualization of the ScheduleDAG.
  std::string getGraphNodeLabel(const llvm::SUnit *SU) const;
  std::string getDAGName() const;
  /// Adds custom features for a visualization of the ScheduleDAG.
  void addCustomGraphFeatures(llvm::GraphWriter<ExpDag *> &) const {}

private:
  template <typename T> void initNodes(const LiveSet &InputLiveReg, T &insts);
  void addDataDep(const llvm::SIRegisterInfo *SIRI);
  void addCtrlDep();
  void buildSubExp(const LiveSet &StartLiveReg, const LiveSet &EndLiveReg,
                   const llvm::SIRegisterInfo *SIRI,
                   const llvm::SIInstrInfo *SIII);
};

struct BlockExpDag : public ExpDag {
  BlockExpDag(llvm::MachineBasicBlock *B, llvm::LiveIntervals *LIS,
              const llvm::MachineRegisterInfo &MRI,
              const llvm::SIRegisterInfo *SIRI, const llvm::SIInstrInfo *SIII);
  llvm::LiveIntervals *LIS;
  llvm::MachineBasicBlock *MBB;
  llvm::DenseMap<llvm::SUnit *, LiveSet> DagPressureMap;
  std::vector<std::vector<llvm::SUnit *>> SUnitsInSameDepth;
  std::vector<SubExp> SubExps;
  void build();
  void buildWithPressure();

private:
  void buildAvail(const LiveSet &PassThruSet,
                  llvm::DenseMap<llvm::SUnit *, LiveSet> &DagAvailRegMap);
  void buildPressure(const LiveSet &StartLiveReg, const LiveSet &EndLiveReg);
};

void getRegBound(llvm::MachineBasicBlock *MBB,
                 const llvm::MachineRegisterInfo &MRI,
                 const llvm::SIRegisterInfo *SIRI,
                 const llvm::SIInstrInfo *SIII, llvm::LiveIntervals *LIS,
                 unsigned &MaxVGPR, unsigned &MaxSGRP);

// Currently mix sgpr and vgpr when build lineage to avoid cycle.
// This maybe waste registers.
// Based on "Minimum Register Instruction Sequencing to Reduce Register Spills
// in Out-of-Order Issue Superscalar Architectures".
class HRB {
public:
  struct Lineage {
    unsigned ID = 0;
    const llvm::TargetRegisterClass *RC = nullptr;
    llvm::SmallVector<llvm::SUnit *, 4> Nodes;
    llvm::SUnit *getHead() const;
    llvm::SUnit *getTail() const;
    void addNode(llvm::SUnit *);
    unsigned getSize() const;
    unsigned length() const;
  };
  struct ColorResult {
    llvm::DenseMap<llvm::SUnit *, unsigned> ColorMap;
    llvm::DenseMap<llvm::SUnit *, unsigned> SizeMap;
    llvm::DenseMap<llvm::SUnit *, unsigned> LineageMap;
    llvm::DenseMap<unsigned, llvm::DenseSet<unsigned>> Conflicts;
    llvm::DenseSet<unsigned> ShareColorLineages;
    llvm::DenseSet<llvm::SUnit *> HeadSet;
    llvm::DenseSet<llvm::SUnit *> TailSet;
    llvm::DenseMap<llvm::SUnit *, llvm::SUnit *> HeadTailMap;
    unsigned maxReg = 0;
    unsigned maxVGPR = 0;
    unsigned maxSGPR = 0;
    void colorSU(llvm::SUnit *SU, unsigned color);
    unsigned getLineage(llvm::SUnit *SU) const;
    bool isConflict(const llvm::SUnit *SU0, unsigned Lineage) const;
    bool isHead(llvm::SUnit *SU) const;
    bool isTail(llvm::SUnit *SU) const;
    const llvm::SUnit *getTail(llvm::SUnit *SU) const;
    unsigned getColor(const llvm::SUnit *SU) const;
    unsigned getSize(const llvm::SUnit *SU) const;
  };
  HRB(const llvm::MachineRegisterInfo &MRI, const llvm::SIRegisterInfo *SIRI)
      : MRI(MRI), SIRI(SIRI) {}

  void buildLinear(std::vector<llvm::SUnit> &SUnits);
  void buildConflict();
  void buildReachRelation(llvm::ArrayRef<llvm::SUnit *> BotRoots);
  llvm::DenseMap<llvm::SUnit *, llvm::DenseSet<llvm::SUnit *>> &getReachMap() {
    return ReachMap;
  }
  bool canReach(llvm::SUnit *a, llvm::SUnit *b);
  void updateReachForEdge(llvm::SUnit *a, llvm::SUnit *b,
                          std::vector<llvm::SUnit> &SUnits);
  void fusionLineages(std::vector<llvm::SUnit> &SUnits);
  ColorResult &coloring();
  void dump();
  void dumpReachMap();

private:
  Lineage buildChain(llvm::SUnit *Node, std::vector<llvm::SUnit> &SUnits);
  llvm::SUnit *findHeir(llvm::SUnit *SU, std::vector<llvm::SUnit> &SUnits);
  bool isConflict(const Lineage &a, const Lineage &b);
  bool canFuse(const Lineage &a, const Lineage &b);
  bool tryFuse(Lineage &a, Lineage &b, std::vector<llvm::SUnit> &SUnits);
  unsigned colorLineages(std::vector<Lineage *> &lineages,
                         llvm::DenseMap<Lineage *, unsigned> &AllocMap,
                         const unsigned Limit);

  llvm::DenseSet<llvm::SUnit *> ChainedNodes;
  llvm::DenseMap<llvm::SUnit *, llvm::DenseSet<llvm::SUnit *>> ReachMap;
  bool IsRecomputeHeight = false;
  std::vector<Lineage> Lineages;
  ColorResult Color;
  const llvm::MachineRegisterInfo &MRI;
  const llvm::SIRegisterInfo *SIRI;
};

std::vector<const llvm::SUnit *> hrbSched(std::vector<llvm::SUnit> &SUnits,
                                          std::vector<llvm::SUnit *> &BRoots,
                                          const llvm::MachineRegisterInfo &MRI,
                                          const llvm::SIRegisterInfo *SIRI);

} // namespace llvm

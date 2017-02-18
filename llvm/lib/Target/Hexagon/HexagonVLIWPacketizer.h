#ifndef HEXAGONVLIWPACKETIZER_H
#define HEXAGONVLIWPACKETIZER_H

#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

namespace llvm {
class HexagonInstrInfo;
class HexagonRegisterInfo;

class HexagonPacketizerList : public VLIWPacketizerList {
  // Vector of instructions assigned to the packet that has just been created.
  std::vector<MachineInstr*> OldPacketMIs;

  // Has the instruction been promoted to a dot-new instruction.
  bool PromotedToDotNew;

  // Has the instruction been glued to allocframe.
  bool GlueAllocframeStore;

  // Has the feeder instruction been glued to new value jump.
  bool GlueToNewValueJump;

  // Check if there is a dependence between some instruction already in this
  // packet and this instruction.
  bool Dependence;

  // Only check for dependence if there are resources available to
  // schedule this instruction.
  bool FoundSequentialDependence;

  // Track MIs with ignored dependence.
  std::vector<MachineInstr*> IgnoreDepMIs;

protected:
  /// \brief A handle to the branch probability pass.
  const MachineBranchProbabilityInfo *MBPI;
  const MachineLoopInfo *MLI;

private:
  const HexagonInstrInfo *HII;
  const HexagonRegisterInfo *HRI;

public:
  // Ctor.
  HexagonPacketizerList(MachineFunction &MF, MachineLoopInfo &MLI,
                        AliasAnalysis *AA,
                        const MachineBranchProbabilityInfo *MBPI);

  // initPacketizerState - initialize some internal flags.
  void initPacketizerState() override;

  // ignorePseudoInstruction - Ignore bundling of pseudo instructions.
  bool ignorePseudoInstruction(const MachineInstr &MI,
                               const MachineBasicBlock *MBB) override;

  // isSoloInstruction - return true if instruction MI can not be packetized
  // with any other instruction, which means that MI itself is a packet.
  bool isSoloInstruction(const MachineInstr &MI) override;

  // isLegalToPacketizeTogether - Is it legal to packetize SUI and SUJ
  // together.
  bool isLegalToPacketizeTogether(SUnit *SUI, SUnit *SUJ) override;

  // isLegalToPruneDependencies - Is it legal to prune dependece between SUI
  // and SUJ.
  bool isLegalToPruneDependencies(SUnit *SUI, SUnit *SUJ) override;

  MachineBasicBlock::iterator addToPacket(MachineInstr &MI) override;
  void endPacket(MachineBasicBlock *MBB,
                 MachineBasicBlock::iterator MI) override;
  bool shouldAddToPacket(const MachineInstr &MI) override;

  void unpacketizeSoloInstrs(MachineFunction &MF);

protected:
  bool isCallDependent(const MachineInstr &MI, SDep::Kind DepType,
                       unsigned DepReg);
  bool promoteToDotCur(MachineInstr &MI, SDep::Kind DepType,
                       MachineBasicBlock::iterator &MII,
                       const TargetRegisterClass *RC);
  bool canPromoteToDotCur(const MachineInstr &MI, const SUnit *PacketSU,
                          unsigned DepReg, MachineBasicBlock::iterator &MII,
                          const TargetRegisterClass *RC);
  void cleanUpDotCur();

  bool promoteToDotNew(MachineInstr &MI, SDep::Kind DepType,
                       MachineBasicBlock::iterator &MII,
                       const TargetRegisterClass *RC);
  bool canPromoteToDotNew(const MachineInstr &MI, const SUnit *PacketSU,
                          unsigned DepReg, MachineBasicBlock::iterator &MII,
                          const TargetRegisterClass *RC);
  bool canPromoteToNewValue(const MachineInstr &MI, const SUnit *PacketSU,
                            unsigned DepReg, MachineBasicBlock::iterator &MII);
  bool canPromoteToNewValueStore(const MachineInstr &MI,
                                 const MachineInstr &PacketMI, unsigned DepReg);
  bool demoteToDotOld(MachineInstr &MI);
  bool useCallersSP(MachineInstr &MI);
  void useCalleesSP(MachineInstr &MI);
  bool arePredicatesComplements(MachineInstr &MI1, MachineInstr &MI2);
  bool restrictingDepExistInPacket(MachineInstr&, unsigned);
  bool isNewifiable(const MachineInstr &MI, const TargetRegisterClass *NewRC);
  bool isCurifiable(MachineInstr &MI);
  bool cannotCoexist(const MachineInstr &MI, const MachineInstr &MJ);
  inline bool isPromotedToDotNew() const {
    return PromotedToDotNew;
  }
  bool tryAllocateResourcesForConstExt(bool Reserve);
  bool canReserveResourcesForConstExt();
  void reserveResourcesForConstExt();
  bool hasDeadDependence(const MachineInstr &I, const MachineInstr &J);
  bool hasControlDependence(const MachineInstr &I, const MachineInstr &J);
  bool hasRegMaskDependence(const MachineInstr &I, const MachineInstr &J);
  bool hasV4SpecificDependence(const MachineInstr &I, const MachineInstr &J);
  bool producesStall(const MachineInstr &MI);
};
} // namespace llvm
#endif // HEXAGONVLIWPACKETIZER_H


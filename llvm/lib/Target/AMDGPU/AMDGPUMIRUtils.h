#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/CodeGen/MachineBasicBlock.h"

namespace llvm {

class MachineFunction;
class LiveIntervals;
class LiveInterval;
class MachineRegisterInfo;
class SIRegisterInfo;
class SIInstrInfo;
class MachineInstr;
class MachinePostDominatorTree;
class MachineLoopInfo;
class MachineDominatorTree;
class raw_ostream;
class TargetInstrInfo;
class TargetRegisterInfo;

typedef unsigned MachineReg;
typedef unsigned MachineOpcode;

constexpr unsigned RegForVCC = 2;
constexpr unsigned VGPR_LIMIT = 256;
// Post RA remat only try to help case when pressue is OK before RA but RA
// result is higher. The diff should not be too much. So just use 4 as threshold
// here.
constexpr unsigned PostRARematThreshHold = 4;

using LiveSet = llvm::DenseMap<unsigned, llvm::LaneBitmask>;

unsigned getRegSize(unsigned Reg, llvm::LaneBitmask &Mask,
                    const llvm::MachineRegisterInfo &MRI,
                    const llvm::SIRegisterInfo *SIRI);
void CollectLiveSetPressure(
    const LiveSet &liveSet,
    const llvm::MachineRegisterInfo &MRI, const llvm::SIRegisterInfo *SIRI,
    unsigned &VPressure, unsigned &SPressure);

bool isExecUpdateForControlFlow(llvm::MachineInstr &MI);

bool IsSub0Sub1SingleDef(unsigned Reg, const llvm::MachineRegisterInfo &MRI);

llvm::LaneBitmask getRegMask(const llvm::MachineOperand &MO,
                             const llvm::MachineRegisterInfo &MRI);
void andLiveRegSet(LiveSet &targetSet, const LiveSet &inputSet);
void andNotLiveRegSet(LiveSet &targetSet, const LiveSet &inputSet);
void mergeLiveRegSet(LiveSet &targetSet, const LiveSet &inputSet);
llvm::MachineBasicBlock *split(llvm::MachineInstr *I);

// For inst like S_BUFFER_LOAD_DWORDX16, change to S_BUFFER_LOAD_DWORDX4 if only
// used 4 lanes.
bool removeUnusedLanes(llvm::MachineInstr &MI, llvm::MachineRegisterInfo &MRI,
                       const llvm::SIRegisterInfo *TRI,
                       const llvm::SIInstrInfo *TII,
                       llvm::SlotIndexes *SlotIndexes);

bool reach_block(llvm::MachineBasicBlock *FromBB, llvm::MachineDominatorTree *DT,
                 llvm::MachinePostDominatorTree *PDT, llvm::MachineLoopInfo *LI,
                 llvm::MachineBasicBlock *ToBB);


void viewCFGWithPhi(llvm::MachineFunction &MF);
void write_contribution_list(llvm::MachineFunction &MF, const char *Filename);

llvm::MachineBasicBlock *CreateNullExportBlock(llvm::MachineFunction &MF, const llvm::SIInstrInfo *TII);

bool GetNonDebugMBBEnd(llvm::MachineBasicBlock::reverse_iterator &BBEnd,
                       llvm::MachineBasicBlock &MBB);

void UpdatePhysRegLiveInForBlock(llvm::MachineBasicBlock *NewBB, const llvm::MachineRegisterInfo *MRI);

void BuildPhysRegLiveInForBlock(llvm::MachineBasicBlock *NewBB,
                                 llvm::SmallDenseSet<unsigned, 8> &LiveOutSet,
                                 const llvm::MachineRegisterInfo *MRI);

MachineReg CreateVirtualRegForOperand(
    MachineOpcode Opcode,
    unsigned Operand,
    llvm::MachineFunction &MF
);

MachineReg CreateVirtualDstReg(
    MachineOpcode Opcode,
    llvm::MachineFunction &MF
);

bool IsExecCopy(const llvm::MachineInstr &MI, MachineReg Exec, MachineReg *pDst);
struct MachineRegWithSubReg {
  MachineReg Reg = AMDGPU::NoRegister;
  unsigned SubReg = AMDGPU::NoSubRegister;
};
MachineRegWithSubReg GetWqmEntryActiveMask(llvm::MachineFunction &MF);
llvm::MachineInstr *GetWqmEntryActiveMaskInst(llvm::MachineFunction &MF);

// Return true if this machine instruction represents a call to the fetch shader.
// We curently have two mechanisims for calling fetch shader:
// 1. The AMDGPU_CALL_FETCH_SHADER pseudo-instruction
// 2. A CALL instruction with the `FetchShaderCall` flag set to true.
bool IsFetchShaderCall(const llvm::MachineInstr* MI);

bool IsSccLiveAt(llvm::MachineBasicBlock *MBB, llvm::MachineBasicBlock::iterator MI);


// An enum used to pass additional constraints to
// `FindOrCreateInsertionPointForSccDef()`. This will further
// constrain the location where the scc def can be inserted.
enum SccDefInsertPointConstraintFlags
{
    None        = 0,   // No additional constraints.
    NoExecWrite = 1,   // Should be no modification of exec between BeforeInst and insert point.
};

// Look for a safe place to insert an instruction that defines scc.
//
//
// This function is useful for when we need to insert a new
// instruction that defines scc in a block and we need to find
// a location that will not smash the existing value.
//
// Starting at `BeforeInst` it will look backwards to try to find
// a place in the block where scc is dead so we can insert our new
// def there. If no location can be found it will save and restore
// scc around BeforeInst. This way BeforeInst can safely be used
// as the new insert location.
//
llvm::MachineBasicBlock::iterator FindOrCreateInsertionPointForSccDef(
    llvm::MachineBasicBlock* MBB,
    llvm::MachineBasicBlock::iterator BeforeInst,
    const llvm::TargetRegisterInfo* TRI,
    const llvm::SIInstrInfo* TII,
    llvm::MachineRegisterInfo* MRI,
    SccDefInsertPointConstraintFlags Constraints = SccDefInsertPointConstraintFlags::None
);

// Check if LI live cross basic blocks, save all touched basic block if is
// local.
bool isLocalLiveInterval(
    const llvm::LiveInterval &LI, llvm::SlotIndexes *Indexes,
    llvm::SmallDenseSet<llvm::MachineBasicBlock *, 2> &touchedMBBSet);
bool isLocalLiveInterval(
    const llvm::LiveInterval &LI, llvm::SlotIndexes *Indexes);

// build liveRegSet at end of each MBB.
void buildEndLiveMap(
    llvm::LiveIntervals *LIS, llvm::MachineFunction &MF,
    const llvm::MachineRegisterInfo &MRI,
    llvm::DenseMap<llvm::MachineBasicBlock *, LiveSet>
        &MBBLiveMap, bool After);

void dumpLiveSet(const LiveSet &LiveSet,
                 const llvm::SIRegisterInfo *SIRI);

unsigned GetCurrentVGPRCount(llvm::MachineFunction &MF, const llvm::SIRegisterInfo *SIRI);
unsigned GetCurrentSGPRCount(llvm::MachineFunction &MF, const llvm::SIRegisterInfo *SIRI);

bool isFastMathInst(llvm::MachineInstr &MI);

namespace pressure {
void print_reg(llvm::Register Reg, const llvm::MachineRegisterInfo &MRI,
               const llvm::SIRegisterInfo *SIRI,
               llvm::raw_ostream &os);
void write_pressure(llvm::MachineFunction &MF, llvm::LiveIntervals *LIS,
                    const char *Filename);
void write_pressure(llvm::MachineFunction &MF, llvm::LiveIntervals *LIS,
                    llvm::raw_ostream &os);
}
// bool IsLdsSpillSupportedForHwStage(xmd::HwStage Stage);

// Look for the successor `Succ` of the given `MBB`.
// Returns MBB->succ_end() if `Succ` is not a successor of MBB.
llvm::MachineBasicBlock::succ_iterator FindSuccessor(llvm::MachineBasicBlock* MBB, llvm::MachineBasicBlock* Succ);

// The enum and helper function for v_perm selection mask.
//
// The input byte layout of v_perm is as below: 
//
// BYTE in[8]
// in[0] = $src1_BYTE0;
// in[1] = $src1_BYTE1;
// in[2] = $src1_BYTE2;
// in[3] = $src1_BYTE3;
// in[4] = $src0_BYTE0;
// in[5] = $src0_BYTE1;
// in[6] = $src0_BYTE2;
// in[7] = $src0_BYTE3;
//
enum class V_PERM_IN_BYTE_POS {
  src1_BYTE0 = 0,
  src1_BYTE1,
  src1_BYTE2,
  src1_BYTE3,
  src0_BYTE0,
  src0_BYTE1,
  src0_BYTE2,
  src0_BYTE3
};

// The 4 arguments specify which input byte will be output
// out[0] = Sel_0;
// out[1] = Sel_1;
// out[2] = Sel_2;
// out[3] = Sel_3;
//
constexpr int buildVPermSelectMask(V_PERM_IN_BYTE_POS Sel_0,
                                   V_PERM_IN_BYTE_POS Sel_1,
                                   V_PERM_IN_BYTE_POS Sel_2,
                                   V_PERM_IN_BYTE_POS Sel_3) {
  return (((int)Sel_3 << 24) | ((int)Sel_2 << 16) |
          ((int)Sel_1 << 8) | (int)Sel_0);
}
}

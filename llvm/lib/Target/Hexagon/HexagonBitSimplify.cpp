//===--- HexagonBitSimplify.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "hexbit"

#include "HexagonBitTracker.h"
#include "HexagonTargetMachine.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace llvm {
  void initializeHexagonBitSimplifyPass(PassRegistry& Registry);
  FunctionPass *createHexagonBitSimplify();
}

namespace {
  // Set of virtual registers, based on BitVector.
  struct RegisterSet : private BitVector {
    RegisterSet() : BitVector() {}
    explicit RegisterSet(unsigned s, bool t = false) : BitVector(s, t) {}
    RegisterSet(const RegisterSet &RS) : BitVector(RS) {}

    using BitVector::clear;
    using BitVector::count;

    unsigned find_first() const {
      int First = BitVector::find_first();
      if (First < 0)
        return 0;
      return x2v(First);
    }

    unsigned find_next(unsigned Prev) const {
      int Next = BitVector::find_next(v2x(Prev));
      if (Next < 0)
        return 0;
      return x2v(Next);
    }

    RegisterSet &insert(unsigned R) {
      unsigned Idx = v2x(R);
      ensure(Idx);
      return static_cast<RegisterSet&>(BitVector::set(Idx));
    }
    RegisterSet &remove(unsigned R) {
      unsigned Idx = v2x(R);
      if (Idx >= size())
        return *this;
      return static_cast<RegisterSet&>(BitVector::reset(Idx));
    }

    RegisterSet &insert(const RegisterSet &Rs) {
      return static_cast<RegisterSet&>(BitVector::operator|=(Rs));
    }
    RegisterSet &remove(const RegisterSet &Rs) {
      return static_cast<RegisterSet&>(BitVector::reset(Rs));
    }

    reference operator[](unsigned R) {
      unsigned Idx = v2x(R);
      ensure(Idx);
      return BitVector::operator[](Idx);
    }
    bool operator[](unsigned R) const {
      unsigned Idx = v2x(R);
      assert(Idx < size());
      return BitVector::operator[](Idx);
    }
    bool has(unsigned R) const {
      unsigned Idx = v2x(R);
      if (Idx >= size())
        return false;
      return BitVector::test(Idx);
    }

    bool empty() const {
      return !BitVector::any();
    }
    bool includes(const RegisterSet &Rs) const {
      // A.BitVector::test(B)  <=>  A-B != {}
      return !Rs.BitVector::test(*this);
    }
    bool intersects(const RegisterSet &Rs) const {
      return BitVector::anyCommon(Rs);
    }

  private:
    void ensure(unsigned Idx) {
      if (size() <= Idx)
        resize(std::max(Idx+1, 32U));
    }
    static inline unsigned v2x(unsigned v) {
      return TargetRegisterInfo::virtReg2Index(v);
    }
    static inline unsigned x2v(unsigned x) {
      return TargetRegisterInfo::index2VirtReg(x);
    }
  };


  struct PrintRegSet {
    PrintRegSet(const RegisterSet &S, const TargetRegisterInfo *RI)
      : RS(S), TRI(RI) {}
    friend raw_ostream &operator<< (raw_ostream &OS,
          const PrintRegSet &P);
  private:
    const RegisterSet &RS;
    const TargetRegisterInfo *TRI;
  };

  raw_ostream &operator<< (raw_ostream &OS, const PrintRegSet &P)
    LLVM_ATTRIBUTE_UNUSED;
  raw_ostream &operator<< (raw_ostream &OS, const PrintRegSet &P) {
    OS << '{';
    for (unsigned R = P.RS.find_first(); R; R = P.RS.find_next(R))
      OS << ' ' << PrintReg(R, P.TRI);
    OS << " }";
    return OS;
  }
}


namespace {
  class Transformation;

  class HexagonBitSimplify : public MachineFunctionPass {
  public:
    static char ID;
    HexagonBitSimplify() : MachineFunctionPass(ID), MDT(0) {
      initializeHexagonBitSimplifyPass(*PassRegistry::getPassRegistry());
    }
    virtual const char *getPassName() const {
      return "Hexagon bit simplification";
    }
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
    virtual bool runOnMachineFunction(MachineFunction &MF);

    static void getInstrDefs(const MachineInstr &MI, RegisterSet &Defs);
    static void getInstrUses(const MachineInstr &MI, RegisterSet &Uses);
    static bool isEqual(const BitTracker::RegisterCell &RC1, uint16_t B1,
        const BitTracker::RegisterCell &RC2, uint16_t B2, uint16_t W);
    static bool isZero(const BitTracker::RegisterCell &RC, uint16_t B,
        uint16_t W);
    static bool getConst(const BitTracker::RegisterCell &RC, uint16_t B,
        uint16_t W, uint64_t &U);
    static bool replaceReg(unsigned OldR, unsigned NewR,
        MachineRegisterInfo &MRI);
    static bool getSubregMask(const BitTracker::RegisterRef &RR,
        unsigned &Begin, unsigned &Width, MachineRegisterInfo &MRI);
    static bool replaceRegWithSub(unsigned OldR, unsigned NewR,
        unsigned NewSR, MachineRegisterInfo &MRI);
    static bool replaceSubWithSub(unsigned OldR, unsigned OldSR,
        unsigned NewR, unsigned NewSR, MachineRegisterInfo &MRI);
    static bool parseRegSequence(const MachineInstr &I,
        BitTracker::RegisterRef &SL, BitTracker::RegisterRef &SH);

    static bool getUsedBitsInStore(unsigned Opc, BitVector &Bits,
        uint16_t Begin);
    static bool getUsedBits(unsigned Opc, unsigned OpN, BitVector &Bits,
        uint16_t Begin, const HexagonInstrInfo &HII);

    static const TargetRegisterClass *getFinalVRegClass(
        const BitTracker::RegisterRef &RR, MachineRegisterInfo &MRI);
    static bool isTransparentCopy(const BitTracker::RegisterRef &RD,
        const BitTracker::RegisterRef &RS, MachineRegisterInfo &MRI);

  private:
    MachineDominatorTree *MDT;

    bool visitBlock(MachineBasicBlock &B, Transformation &T, RegisterSet &AVs);
  };

  char HexagonBitSimplify::ID = 0;
  typedef HexagonBitSimplify HBS;


  // The purpose of this class is to provide a common facility to traverse
  // the function top-down or bottom-up via the dominator tree, and keep
  // track of the available registers.
  class Transformation {
  public:
    bool TopDown;
    Transformation(bool TD) : TopDown(TD) {}
    virtual bool processBlock(MachineBasicBlock &B, const RegisterSet &AVs) = 0;
    virtual ~Transformation() {}
  };
}

INITIALIZE_PASS_BEGIN(HexagonBitSimplify, "hexbit",
      "Hexagon bit simplification", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(HexagonBitSimplify, "hexbit",
      "Hexagon bit simplification", false, false)


bool HexagonBitSimplify::visitBlock(MachineBasicBlock &B, Transformation &T,
      RegisterSet &AVs) {
  MachineDomTreeNode *N = MDT->getNode(&B);
  typedef GraphTraits<MachineDomTreeNode*> GTN;
  bool Changed = false;

  if (T.TopDown)
    Changed = T.processBlock(B, AVs);

  RegisterSet Defs;
  for (auto &I : B)
    getInstrDefs(I, Defs);
  RegisterSet NewAVs = AVs;
  NewAVs.insert(Defs);

  for (auto I = GTN::child_begin(N), E = GTN::child_end(N); I != E; ++I) {
    MachineBasicBlock *SB = (*I)->getBlock();
    Changed |= visitBlock(*SB, T, NewAVs);
  }
  if (!T.TopDown)
    Changed |= T.processBlock(B, AVs);

  return Changed;
}

//
// Utility functions:
//
void HexagonBitSimplify::getInstrDefs(const MachineInstr &MI,
      RegisterSet &Defs) {
  for (auto &Op : MI.operands()) {
    if (!Op.isReg() || !Op.isDef())
      continue;
    unsigned R = Op.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(R))
      continue;
    Defs.insert(R);
  }
}

void HexagonBitSimplify::getInstrUses(const MachineInstr &MI,
      RegisterSet &Uses) {
  for (auto &Op : MI.operands()) {
    if (!Op.isReg() || !Op.isUse())
      continue;
    unsigned R = Op.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(R))
      continue;
    Uses.insert(R);
  }
}

// Check if all the bits in range [B, E) in both cells are equal.
bool HexagonBitSimplify::isEqual(const BitTracker::RegisterCell &RC1,
      uint16_t B1, const BitTracker::RegisterCell &RC2, uint16_t B2,
      uint16_t W) {
  for (uint16_t i = 0; i < W; ++i) {
    // If RC1[i] is "bottom", it cannot be proven equal to RC2[i].
    if (RC1[B1+i].Type == BitTracker::BitValue::Ref && RC1[B1+i].RefI.Reg == 0)
      return false;
    // Same for RC2[i].
    if (RC2[B2+i].Type == BitTracker::BitValue::Ref && RC2[B2+i].RefI.Reg == 0)
      return false;
    if (RC1[B1+i] != RC2[B2+i])
      return false;
  }
  return true;
}

bool HexagonBitSimplify::isZero(const BitTracker::RegisterCell &RC,
      uint16_t B, uint16_t W) {
  assert(B < RC.width() && B+W <= RC.width());
  for (uint16_t i = B; i < B+W; ++i)
    if (!RC[i].is(0))
      return false;
  return true;
}


bool HexagonBitSimplify::getConst(const BitTracker::RegisterCell &RC,
        uint16_t B, uint16_t W, uint64_t &U) {
  assert(B < RC.width() && B+W <= RC.width());
  int64_t T = 0;
  for (uint16_t i = B+W; i > B; --i) {
    const BitTracker::BitValue &BV = RC[i-1];
    T <<= 1;
    if (BV.is(1))
      T |= 1;
    else if (!BV.is(0))
      return false;
  }
  U = T;
  return true;
}


bool HexagonBitSimplify::replaceReg(unsigned OldR, unsigned NewR,
      MachineRegisterInfo &MRI) {
  if (!TargetRegisterInfo::isVirtualRegister(OldR) ||
      !TargetRegisterInfo::isVirtualRegister(NewR))
    return false;
  auto Begin = MRI.use_begin(OldR), End = MRI.use_end();
  decltype(End) NextI;
  for (auto I = Begin; I != End; I = NextI) {
    NextI = std::next(I);
    I->setReg(NewR);
  }
  return Begin != End;
}


bool HexagonBitSimplify::replaceRegWithSub(unsigned OldR, unsigned NewR,
      unsigned NewSR, MachineRegisterInfo &MRI) {
  if (!TargetRegisterInfo::isVirtualRegister(OldR) ||
      !TargetRegisterInfo::isVirtualRegister(NewR))
    return false;
  auto Begin = MRI.use_begin(OldR), End = MRI.use_end();
  decltype(End) NextI;
  for (auto I = Begin; I != End; I = NextI) {
    NextI = std::next(I);
    I->setReg(NewR);
    I->setSubReg(NewSR);
  }
  return Begin != End;
}


bool HexagonBitSimplify::replaceSubWithSub(unsigned OldR, unsigned OldSR,
      unsigned NewR, unsigned NewSR, MachineRegisterInfo &MRI) {
  if (!TargetRegisterInfo::isVirtualRegister(OldR) ||
      !TargetRegisterInfo::isVirtualRegister(NewR))
    return false;
  auto Begin = MRI.use_begin(OldR), End = MRI.use_end();
  decltype(End) NextI;
  for (auto I = Begin; I != End; I = NextI) {
    NextI = std::next(I);
    if (I->getSubReg() != OldSR)
      continue;
    I->setReg(NewR);
    I->setSubReg(NewSR);
  }
  return Begin != End;
}


// For a register ref (pair Reg:Sub), set Begin to the position of the LSB
// of Sub in Reg, and set Width to the size of Sub in bits. Return true,
// if this succeeded, otherwise return false.
bool HexagonBitSimplify::getSubregMask(const BitTracker::RegisterRef &RR,
      unsigned &Begin, unsigned &Width, MachineRegisterInfo &MRI) {
  const TargetRegisterClass *RC = MRI.getRegClass(RR.Reg);
  if (RC == &Hexagon::IntRegsRegClass) {
    assert(RR.Sub == 0);
    Begin = 0;
    Width = 32;
    return true;
  }
  if (RC == &Hexagon::DoubleRegsRegClass) {
    if (RR.Sub == 0) {
      Begin = 0;
      Width = 64;
      return true;
    }
    assert(RR.Sub == Hexagon::subreg_loreg || RR.Sub == Hexagon::subreg_hireg);
    Width = 32;
    Begin = (RR.Sub == Hexagon::subreg_loreg ? 0 : 32);
    return true;
  }
  return false;
}


// For a REG_SEQUENCE, set SL to the low subregister and SH to the high
// subregister.
bool HexagonBitSimplify::parseRegSequence(const MachineInstr &I,
      BitTracker::RegisterRef &SL, BitTracker::RegisterRef &SH) {
  assert(I.getOpcode() == TargetOpcode::REG_SEQUENCE);
  unsigned Sub1 = I.getOperand(2).getImm(), Sub2 = I.getOperand(4).getImm();
  assert(Sub1 != Sub2);
  if (Sub1 == Hexagon::subreg_loreg && Sub2 == Hexagon::subreg_hireg) {
    SL = I.getOperand(1);
    SH = I.getOperand(3);
    return true;
  }
  if (Sub1 == Hexagon::subreg_hireg && Sub2 == Hexagon::subreg_loreg) {
    SH = I.getOperand(1);
    SL = I.getOperand(3);
    return true;
  }
  return false;
}


// All stores (except 64-bit stores) take a 32-bit register as the source
// of the value to be stored. If the instruction stores into a location
// that is shorter than 32 bits, some bits of the source register are not
// used. For each store instruction, calculate the set of used bits in
// the source register, and set appropriate bits in Bits. Return true if
// the bits are calculated, false otherwise.
bool HexagonBitSimplify::getUsedBitsInStore(unsigned Opc, BitVector &Bits,
      uint16_t Begin) {
  using namespace Hexagon;

  switch (Opc) {
    // Store byte
    case S2_storerb_io:           // memb(Rs32+#s11:0)=Rt32
    case S2_storerbnew_io:        // memb(Rs32+#s11:0)=Nt8.new
    case S2_pstorerbt_io:         // if (Pv4) memb(Rs32+#u6:0)=Rt32
    case S2_pstorerbf_io:         // if (!Pv4) memb(Rs32+#u6:0)=Rt32
    case S4_pstorerbtnew_io:      // if (Pv4.new) memb(Rs32+#u6:0)=Rt32
    case S4_pstorerbfnew_io:      // if (!Pv4.new) memb(Rs32+#u6:0)=Rt32
    case S2_pstorerbnewt_io:      // if (Pv4) memb(Rs32+#u6:0)=Nt8.new
    case S2_pstorerbnewf_io:      // if (!Pv4) memb(Rs32+#u6:0)=Nt8.new
    case S4_pstorerbnewtnew_io:   // if (Pv4.new) memb(Rs32+#u6:0)=Nt8.new
    case S4_pstorerbnewfnew_io:   // if (!Pv4.new) memb(Rs32+#u6:0)=Nt8.new
    case S2_storerb_pi:           // memb(Rx32++#s4:0)=Rt32
    case S2_storerbnew_pi:        // memb(Rx32++#s4:0)=Nt8.new
    case S2_pstorerbt_pi:         // if (Pv4) memb(Rx32++#s4:0)=Rt32
    case S2_pstorerbf_pi:         // if (!Pv4) memb(Rx32++#s4:0)=Rt32
    case S2_pstorerbtnew_pi:      // if (Pv4.new) memb(Rx32++#s4:0)=Rt32
    case S2_pstorerbfnew_pi:      // if (!Pv4.new) memb(Rx32++#s4:0)=Rt32
    case S2_pstorerbnewt_pi:      // if (Pv4) memb(Rx32++#s4:0)=Nt8.new
    case S2_pstorerbnewf_pi:      // if (!Pv4) memb(Rx32++#s4:0)=Nt8.new
    case S2_pstorerbnewtnew_pi:   // if (Pv4.new) memb(Rx32++#s4:0)=Nt8.new
    case S2_pstorerbnewfnew_pi:   // if (!Pv4.new) memb(Rx32++#s4:0)=Nt8.new
    case S4_storerb_ap:           // memb(Re32=#U6)=Rt32
    case S4_storerbnew_ap:        // memb(Re32=#U6)=Nt8.new
    case S2_storerb_pr:           // memb(Rx32++Mu2)=Rt32
    case S2_storerbnew_pr:        // memb(Rx32++Mu2)=Nt8.new
    case S4_storerb_ur:           // memb(Ru32<<#u2+#U6)=Rt32
    case S4_storerbnew_ur:        // memb(Ru32<<#u2+#U6)=Nt8.new
    case S2_storerb_pbr:          // memb(Rx32++Mu2:brev)=Rt32
    case S2_storerbnew_pbr:       // memb(Rx32++Mu2:brev)=Nt8.new
    case S2_storerb_pci:          // memb(Rx32++#s4:0:circ(Mu2))=Rt32
    case S2_storerbnew_pci:       // memb(Rx32++#s4:0:circ(Mu2))=Nt8.new
    case S2_storerb_pcr:          // memb(Rx32++I:circ(Mu2))=Rt32
    case S2_storerbnew_pcr:       // memb(Rx32++I:circ(Mu2))=Nt8.new
    case S4_storerb_rr:           // memb(Rs32+Ru32<<#u2)=Rt32
    case S4_storerbnew_rr:        // memb(Rs32+Ru32<<#u2)=Nt8.new
    case S4_pstorerbt_rr:         // if (Pv4) memb(Rs32+Ru32<<#u2)=Rt32
    case S4_pstorerbf_rr:         // if (!Pv4) memb(Rs32+Ru32<<#u2)=Rt32
    case S4_pstorerbtnew_rr:      // if (Pv4.new) memb(Rs32+Ru32<<#u2)=Rt32
    case S4_pstorerbfnew_rr:      // if (!Pv4.new) memb(Rs32+Ru32<<#u2)=Rt32
    case S4_pstorerbnewt_rr:      // if (Pv4) memb(Rs32+Ru32<<#u2)=Nt8.new
    case S4_pstorerbnewf_rr:      // if (!Pv4) memb(Rs32+Ru32<<#u2)=Nt8.new
    case S4_pstorerbnewtnew_rr:   // if (Pv4.new) memb(Rs32+Ru32<<#u2)=Nt8.new
    case S4_pstorerbnewfnew_rr:   // if (!Pv4.new) memb(Rs32+Ru32<<#u2)=Nt8.new
    case S2_storerbgp:            // memb(gp+#u16:0)=Rt32
    case S2_storerbnewgp:         // memb(gp+#u16:0)=Nt8.new
    case S4_pstorerbt_abs:        // if (Pv4) memb(#u6)=Rt32
    case S4_pstorerbf_abs:        // if (!Pv4) memb(#u6)=Rt32
    case S4_pstorerbtnew_abs:     // if (Pv4.new) memb(#u6)=Rt32
    case S4_pstorerbfnew_abs:     // if (!Pv4.new) memb(#u6)=Rt32
    case S4_pstorerbnewt_abs:     // if (Pv4) memb(#u6)=Nt8.new
    case S4_pstorerbnewf_abs:     // if (!Pv4) memb(#u6)=Nt8.new
    case S4_pstorerbnewtnew_abs:  // if (Pv4.new) memb(#u6)=Nt8.new
    case S4_pstorerbnewfnew_abs:  // if (!Pv4.new) memb(#u6)=Nt8.new
      Bits.set(Begin, Begin+8);
      return true;

    // Store low half
    case S2_storerh_io:           // memh(Rs32+#s11:1)=Rt32
    case S2_storerhnew_io:        // memh(Rs32+#s11:1)=Nt8.new
    case S2_pstorerht_io:         // if (Pv4) memh(Rs32+#u6:1)=Rt32
    case S2_pstorerhf_io:         // if (!Pv4) memh(Rs32+#u6:1)=Rt32
    case S4_pstorerhtnew_io:      // if (Pv4.new) memh(Rs32+#u6:1)=Rt32
    case S4_pstorerhfnew_io:      // if (!Pv4.new) memh(Rs32+#u6:1)=Rt32
    case S2_pstorerhnewt_io:      // if (Pv4) memh(Rs32+#u6:1)=Nt8.new
    case S2_pstorerhnewf_io:      // if (!Pv4) memh(Rs32+#u6:1)=Nt8.new
    case S4_pstorerhnewtnew_io:   // if (Pv4.new) memh(Rs32+#u6:1)=Nt8.new
    case S4_pstorerhnewfnew_io:   // if (!Pv4.new) memh(Rs32+#u6:1)=Nt8.new
    case S2_storerh_pi:           // memh(Rx32++#s4:1)=Rt32
    case S2_storerhnew_pi:        // memh(Rx32++#s4:1)=Nt8.new
    case S2_pstorerht_pi:         // if (Pv4) memh(Rx32++#s4:1)=Rt32
    case S2_pstorerhf_pi:         // if (!Pv4) memh(Rx32++#s4:1)=Rt32
    case S2_pstorerhtnew_pi:      // if (Pv4.new) memh(Rx32++#s4:1)=Rt32
    case S2_pstorerhfnew_pi:      // if (!Pv4.new) memh(Rx32++#s4:1)=Rt32
    case S2_pstorerhnewt_pi:      // if (Pv4) memh(Rx32++#s4:1)=Nt8.new
    case S2_pstorerhnewf_pi:      // if (!Pv4) memh(Rx32++#s4:1)=Nt8.new
    case S2_pstorerhnewtnew_pi:   // if (Pv4.new) memh(Rx32++#s4:1)=Nt8.new
    case S2_pstorerhnewfnew_pi:   // if (!Pv4.new) memh(Rx32++#s4:1)=Nt8.new
    case S4_storerh_ap:           // memh(Re32=#U6)=Rt32
    case S4_storerhnew_ap:        // memh(Re32=#U6)=Nt8.new
    case S2_storerh_pr:           // memh(Rx32++Mu2)=Rt32
    case S2_storerhnew_pr:        // memh(Rx32++Mu2)=Nt8.new
    case S4_storerh_ur:           // memh(Ru32<<#u2+#U6)=Rt32
    case S4_storerhnew_ur:        // memh(Ru32<<#u2+#U6)=Nt8.new
    case S2_storerh_pbr:          // memh(Rx32++Mu2:brev)=Rt32
    case S2_storerhnew_pbr:       // memh(Rx32++Mu2:brev)=Nt8.new
    case S2_storerh_pci:          // memh(Rx32++#s4:1:circ(Mu2))=Rt32
    case S2_storerhnew_pci:       // memh(Rx32++#s4:1:circ(Mu2))=Nt8.new
    case S2_storerh_pcr:          // memh(Rx32++I:circ(Mu2))=Rt32
    case S2_storerhnew_pcr:       // memh(Rx32++I:circ(Mu2))=Nt8.new
    case S4_storerh_rr:           // memh(Rs32+Ru32<<#u2)=Rt32
    case S4_pstorerht_rr:         // if (Pv4) memh(Rs32+Ru32<<#u2)=Rt32
    case S4_pstorerhf_rr:         // if (!Pv4) memh(Rs32+Ru32<<#u2)=Rt32
    case S4_pstorerhtnew_rr:      // if (Pv4.new) memh(Rs32+Ru32<<#u2)=Rt32
    case S4_pstorerhfnew_rr:      // if (!Pv4.new) memh(Rs32+Ru32<<#u2)=Rt32
    case S4_storerhnew_rr:        // memh(Rs32+Ru32<<#u2)=Nt8.new
    case S4_pstorerhnewt_rr:      // if (Pv4) memh(Rs32+Ru32<<#u2)=Nt8.new
    case S4_pstorerhnewf_rr:      // if (!Pv4) memh(Rs32+Ru32<<#u2)=Nt8.new
    case S4_pstorerhnewtnew_rr:   // if (Pv4.new) memh(Rs32+Ru32<<#u2)=Nt8.new
    case S4_pstorerhnewfnew_rr:   // if (!Pv4.new) memh(Rs32+Ru32<<#u2)=Nt8.new
    case S2_storerhgp:            // memh(gp+#u16:1)=Rt32
    case S2_storerhnewgp:         // memh(gp+#u16:1)=Nt8.new
    case S4_pstorerht_abs:        // if (Pv4) memh(#u6)=Rt32
    case S4_pstorerhf_abs:        // if (!Pv4) memh(#u6)=Rt32
    case S4_pstorerhtnew_abs:     // if (Pv4.new) memh(#u6)=Rt32
    case S4_pstorerhfnew_abs:     // if (!Pv4.new) memh(#u6)=Rt32
    case S4_pstorerhnewt_abs:     // if (Pv4) memh(#u6)=Nt8.new
    case S4_pstorerhnewf_abs:     // if (!Pv4) memh(#u6)=Nt8.new
    case S4_pstorerhnewtnew_abs:  // if (Pv4.new) memh(#u6)=Nt8.new
    case S4_pstorerhnewfnew_abs:  // if (!Pv4.new) memh(#u6)=Nt8.new
      Bits.set(Begin, Begin+16);
      return true;

    // Store high half
    case S2_storerf_io:           // memh(Rs32+#s11:1)=Rt.H32
    case S2_pstorerft_io:         // if (Pv4) memh(Rs32+#u6:1)=Rt.H32
    case S2_pstorerff_io:         // if (!Pv4) memh(Rs32+#u6:1)=Rt.H32
    case S4_pstorerftnew_io:      // if (Pv4.new) memh(Rs32+#u6:1)=Rt.H32
    case S4_pstorerffnew_io:      // if (!Pv4.new) memh(Rs32+#u6:1)=Rt.H32
    case S2_storerf_pi:           // memh(Rx32++#s4:1)=Rt.H32
    case S2_pstorerft_pi:         // if (Pv4) memh(Rx32++#s4:1)=Rt.H32
    case S2_pstorerff_pi:         // if (!Pv4) memh(Rx32++#s4:1)=Rt.H32
    case S2_pstorerftnew_pi:      // if (Pv4.new) memh(Rx32++#s4:1)=Rt.H32
    case S2_pstorerffnew_pi:      // if (!Pv4.new) memh(Rx32++#s4:1)=Rt.H32
    case S4_storerf_ap:           // memh(Re32=#U6)=Rt.H32
    case S2_storerf_pr:           // memh(Rx32++Mu2)=Rt.H32
    case S4_storerf_ur:           // memh(Ru32<<#u2+#U6)=Rt.H32
    case S2_storerf_pbr:          // memh(Rx32++Mu2:brev)=Rt.H32
    case S2_storerf_pci:          // memh(Rx32++#s4:1:circ(Mu2))=Rt.H32
    case S2_storerf_pcr:          // memh(Rx32++I:circ(Mu2))=Rt.H32
    case S4_storerf_rr:           // memh(Rs32+Ru32<<#u2)=Rt.H32
    case S4_pstorerft_rr:         // if (Pv4) memh(Rs32+Ru32<<#u2)=Rt.H32
    case S4_pstorerff_rr:         // if (!Pv4) memh(Rs32+Ru32<<#u2)=Rt.H32
    case S4_pstorerftnew_rr:      // if (Pv4.new) memh(Rs32+Ru32<<#u2)=Rt.H32
    case S4_pstorerffnew_rr:      // if (!Pv4.new) memh(Rs32+Ru32<<#u2)=Rt.H32
    case S2_storerfgp:            // memh(gp+#u16:1)=Rt.H32
    case S4_pstorerft_abs:        // if (Pv4) memh(#u6)=Rt.H32
    case S4_pstorerff_abs:        // if (!Pv4) memh(#u6)=Rt.H32
    case S4_pstorerftnew_abs:     // if (Pv4.new) memh(#u6)=Rt.H32
    case S4_pstorerffnew_abs:     // if (!Pv4.new) memh(#u6)=Rt.H32
      Bits.set(Begin+16, Begin+32);
      return true;
  }

  return false;
}


// For an instruction with opcode Opc, calculate the set of bits that it
// uses in a register in operand OpN. This only calculates the set of used
// bits for cases where it does not depend on any operands (as is the case
// in shifts, for example). For concrete instructions from a program, the
// operand may be a subregister of a larger register, while Bits would
// correspond to the larger register in its entirety. Because of that,
// the parameter Begin can be used to indicate which bit of Bits should be
// considered the LSB of of the operand.
bool HexagonBitSimplify::getUsedBits(unsigned Opc, unsigned OpN,
      BitVector &Bits, uint16_t Begin, const HexagonInstrInfo &HII) {
  using namespace Hexagon;

  const MCInstrDesc &D = HII.get(Opc);
  if (D.mayStore()) {
    if (OpN == D.getNumOperands()-1)
      return getUsedBitsInStore(Opc, Bits, Begin);
    return false;
  }

  switch (Opc) {
    // One register source. Used bits: R1[0-7].
    case A2_sxtb:
    case A2_zxtb:
    case A4_cmpbeqi:
    case A4_cmpbgti:
    case A4_cmpbgtui:
      if (OpN == 1) {
        Bits.set(Begin, Begin+8);
        return true;
      }
      break;

    // One register source. Used bits: R1[0-15].
    case A2_aslh:
    case A2_sxth:
    case A2_zxth:
    case A4_cmpheqi:
    case A4_cmphgti:
    case A4_cmphgtui:
      if (OpN == 1) {
        Bits.set(Begin, Begin+16);
        return true;
      }
      break;

    // One register source. Used bits: R1[16-31].
    case A2_asrh:
      if (OpN == 1) {
        Bits.set(Begin+16, Begin+32);
        return true;
      }
      break;

    // Two register sources. Used bits: R1[0-7], R2[0-7].
    case A4_cmpbeq:
    case A4_cmpbgt:
    case A4_cmpbgtu:
      if (OpN == 1) {
        Bits.set(Begin, Begin+8);
        return true;
      }
      break;

    // Two register sources. Used bits: R1[0-15], R2[0-15].
    case A4_cmpheq:
    case A4_cmphgt:
    case A4_cmphgtu:
    case A2_addh_h16_ll:
    case A2_addh_h16_sat_ll:
    case A2_addh_l16_ll:
    case A2_addh_l16_sat_ll:
    case A2_combine_ll:
    case A2_subh_h16_ll:
    case A2_subh_h16_sat_ll:
    case A2_subh_l16_ll:
    case A2_subh_l16_sat_ll:
    case M2_mpy_acc_ll_s0:
    case M2_mpy_acc_ll_s1:
    case M2_mpy_acc_sat_ll_s0:
    case M2_mpy_acc_sat_ll_s1:
    case M2_mpy_ll_s0:
    case M2_mpy_ll_s1:
    case M2_mpy_nac_ll_s0:
    case M2_mpy_nac_ll_s1:
    case M2_mpy_nac_sat_ll_s0:
    case M2_mpy_nac_sat_ll_s1:
    case M2_mpy_rnd_ll_s0:
    case M2_mpy_rnd_ll_s1:
    case M2_mpy_sat_ll_s0:
    case M2_mpy_sat_ll_s1:
    case M2_mpy_sat_rnd_ll_s0:
    case M2_mpy_sat_rnd_ll_s1:
    case M2_mpyd_acc_ll_s0:
    case M2_mpyd_acc_ll_s1:
    case M2_mpyd_ll_s0:
    case M2_mpyd_ll_s1:
    case M2_mpyd_nac_ll_s0:
    case M2_mpyd_nac_ll_s1:
    case M2_mpyd_rnd_ll_s0:
    case M2_mpyd_rnd_ll_s1:
    case M2_mpyu_acc_ll_s0:
    case M2_mpyu_acc_ll_s1:
    case M2_mpyu_ll_s0:
    case M2_mpyu_ll_s1:
    case M2_mpyu_nac_ll_s0:
    case M2_mpyu_nac_ll_s1:
    case M2_mpyud_acc_ll_s0:
    case M2_mpyud_acc_ll_s1:
    case M2_mpyud_ll_s0:
    case M2_mpyud_ll_s1:
    case M2_mpyud_nac_ll_s0:
    case M2_mpyud_nac_ll_s1:
      if (OpN == 1 || OpN == 2) {
        Bits.set(Begin, Begin+16);
        return true;
      }
      break;

    // Two register sources. Used bits: R1[0-15], R2[16-31].
    case A2_addh_h16_lh:
    case A2_addh_h16_sat_lh:
    case A2_combine_lh:
    case A2_subh_h16_lh:
    case A2_subh_h16_sat_lh:
    case M2_mpy_acc_lh_s0:
    case M2_mpy_acc_lh_s1:
    case M2_mpy_acc_sat_lh_s0:
    case M2_mpy_acc_sat_lh_s1:
    case M2_mpy_lh_s0:
    case M2_mpy_lh_s1:
    case M2_mpy_nac_lh_s0:
    case M2_mpy_nac_lh_s1:
    case M2_mpy_nac_sat_lh_s0:
    case M2_mpy_nac_sat_lh_s1:
    case M2_mpy_rnd_lh_s0:
    case M2_mpy_rnd_lh_s1:
    case M2_mpy_sat_lh_s0:
    case M2_mpy_sat_lh_s1:
    case M2_mpy_sat_rnd_lh_s0:
    case M2_mpy_sat_rnd_lh_s1:
    case M2_mpyd_acc_lh_s0:
    case M2_mpyd_acc_lh_s1:
    case M2_mpyd_lh_s0:
    case M2_mpyd_lh_s1:
    case M2_mpyd_nac_lh_s0:
    case M2_mpyd_nac_lh_s1:
    case M2_mpyd_rnd_lh_s0:
    case M2_mpyd_rnd_lh_s1:
    case M2_mpyu_acc_lh_s0:
    case M2_mpyu_acc_lh_s1:
    case M2_mpyu_lh_s0:
    case M2_mpyu_lh_s1:
    case M2_mpyu_nac_lh_s0:
    case M2_mpyu_nac_lh_s1:
    case M2_mpyud_acc_lh_s0:
    case M2_mpyud_acc_lh_s1:
    case M2_mpyud_lh_s0:
    case M2_mpyud_lh_s1:
    case M2_mpyud_nac_lh_s0:
    case M2_mpyud_nac_lh_s1:
    // These four are actually LH.
    case A2_addh_l16_hl:
    case A2_addh_l16_sat_hl:
    case A2_subh_l16_hl:
    case A2_subh_l16_sat_hl:
      if (OpN == 1) {
        Bits.set(Begin, Begin+16);
        return true;
      }
      if (OpN == 2) {
        Bits.set(Begin+16, Begin+32);
        return true;
      }
      break;

    // Two register sources, used bits: R1[16-31], R2[0-15].
    case A2_addh_h16_hl:
    case A2_addh_h16_sat_hl:
    case A2_combine_hl:
    case A2_subh_h16_hl:
    case A2_subh_h16_sat_hl:
    case M2_mpy_acc_hl_s0:
    case M2_mpy_acc_hl_s1:
    case M2_mpy_acc_sat_hl_s0:
    case M2_mpy_acc_sat_hl_s1:
    case M2_mpy_hl_s0:
    case M2_mpy_hl_s1:
    case M2_mpy_nac_hl_s0:
    case M2_mpy_nac_hl_s1:
    case M2_mpy_nac_sat_hl_s0:
    case M2_mpy_nac_sat_hl_s1:
    case M2_mpy_rnd_hl_s0:
    case M2_mpy_rnd_hl_s1:
    case M2_mpy_sat_hl_s0:
    case M2_mpy_sat_hl_s1:
    case M2_mpy_sat_rnd_hl_s0:
    case M2_mpy_sat_rnd_hl_s1:
    case M2_mpyd_acc_hl_s0:
    case M2_mpyd_acc_hl_s1:
    case M2_mpyd_hl_s0:
    case M2_mpyd_hl_s1:
    case M2_mpyd_nac_hl_s0:
    case M2_mpyd_nac_hl_s1:
    case M2_mpyd_rnd_hl_s0:
    case M2_mpyd_rnd_hl_s1:
    case M2_mpyu_acc_hl_s0:
    case M2_mpyu_acc_hl_s1:
    case M2_mpyu_hl_s0:
    case M2_mpyu_hl_s1:
    case M2_mpyu_nac_hl_s0:
    case M2_mpyu_nac_hl_s1:
    case M2_mpyud_acc_hl_s0:
    case M2_mpyud_acc_hl_s1:
    case M2_mpyud_hl_s0:
    case M2_mpyud_hl_s1:
    case M2_mpyud_nac_hl_s0:
    case M2_mpyud_nac_hl_s1:
      if (OpN == 1) {
        Bits.set(Begin+16, Begin+32);
        return true;
      }
      if (OpN == 2) {
        Bits.set(Begin, Begin+16);
        return true;
      }
      break;

    // Two register sources, used bits: R1[16-31], R2[16-31].
    case A2_addh_h16_hh:
    case A2_addh_h16_sat_hh:
    case A2_combine_hh:
    case A2_subh_h16_hh:
    case A2_subh_h16_sat_hh:
    case M2_mpy_acc_hh_s0:
    case M2_mpy_acc_hh_s1:
    case M2_mpy_acc_sat_hh_s0:
    case M2_mpy_acc_sat_hh_s1:
    case M2_mpy_hh_s0:
    case M2_mpy_hh_s1:
    case M2_mpy_nac_hh_s0:
    case M2_mpy_nac_hh_s1:
    case M2_mpy_nac_sat_hh_s0:
    case M2_mpy_nac_sat_hh_s1:
    case M2_mpy_rnd_hh_s0:
    case M2_mpy_rnd_hh_s1:
    case M2_mpy_sat_hh_s0:
    case M2_mpy_sat_hh_s1:
    case M2_mpy_sat_rnd_hh_s0:
    case M2_mpy_sat_rnd_hh_s1:
    case M2_mpyd_acc_hh_s0:
    case M2_mpyd_acc_hh_s1:
    case M2_mpyd_hh_s0:
    case M2_mpyd_hh_s1:
    case M2_mpyd_nac_hh_s0:
    case M2_mpyd_nac_hh_s1:
    case M2_mpyd_rnd_hh_s0:
    case M2_mpyd_rnd_hh_s1:
    case M2_mpyu_acc_hh_s0:
    case M2_mpyu_acc_hh_s1:
    case M2_mpyu_hh_s0:
    case M2_mpyu_hh_s1:
    case M2_mpyu_nac_hh_s0:
    case M2_mpyu_nac_hh_s1:
    case M2_mpyud_acc_hh_s0:
    case M2_mpyud_acc_hh_s1:
    case M2_mpyud_hh_s0:
    case M2_mpyud_hh_s1:
    case M2_mpyud_nac_hh_s0:
    case M2_mpyud_nac_hh_s1:
      if (OpN == 1 || OpN == 2) {
        Bits.set(Begin+16, Begin+32);
        return true;
      }
      break;
  }

  return false;
}


// Calculate the register class that matches Reg:Sub. For example, if
// vreg1 is a double register, then vreg1:subreg_hireg would match "int"
// register class.
const TargetRegisterClass *HexagonBitSimplify::getFinalVRegClass(
      const BitTracker::RegisterRef &RR, MachineRegisterInfo &MRI) {
  if (!TargetRegisterInfo::isVirtualRegister(RR.Reg))
    return nullptr;
  auto *RC = MRI.getRegClass(RR.Reg);
  if (RR.Sub == 0)
    return RC;

  auto VerifySR = [] (unsigned Sub) -> void {
    assert(Sub == Hexagon::subreg_hireg || Sub == Hexagon::subreg_loreg);
  };

  switch (RC->getID()) {
    case Hexagon::DoubleRegsRegClassID:
      VerifySR(RR.Sub);
      return &Hexagon::IntRegsRegClass;
    case Hexagon::VecDblRegsRegClassID:
      VerifySR(RR.Sub);
      return &Hexagon::VectorRegsRegClass;
    case Hexagon::VecDblRegs128BRegClassID:
      VerifySR(RR.Sub);
      return &Hexagon::VectorRegs128BRegClass;
  }
  return nullptr;
}


// Check if RD could be replaced with RS at any possible use of RD.
// For example a predicate register cannot be replaced with a integer
// register, but a 64-bit register with a subregister can be replaced
// with a 32-bit register.
bool HexagonBitSimplify::isTransparentCopy(const BitTracker::RegisterRef &RD,
      const BitTracker::RegisterRef &RS, MachineRegisterInfo &MRI) {
  if (!TargetRegisterInfo::isVirtualRegister(RD.Reg) ||
      !TargetRegisterInfo::isVirtualRegister(RS.Reg))
    return false;
  // Return false if one (or both) classes are nullptr.
  auto *DRC = getFinalVRegClass(RD, MRI);
  if (!DRC)
    return false;

  return DRC == getFinalVRegClass(RS, MRI);
}


//
// Dead code elimination
//
namespace {
  class DeadCodeElimination {
  public:
    DeadCodeElimination(MachineFunction &mf, MachineDominatorTree &mdt)
      : MF(mf), HII(*MF.getSubtarget<HexagonSubtarget>().getInstrInfo()),
        MDT(mdt), MRI(mf.getRegInfo()) {}

    bool run() {
      return runOnNode(MDT.getRootNode());
    }

  private:
    bool isDead(unsigned R) const;
    bool runOnNode(MachineDomTreeNode *N);

    MachineFunction &MF;
    const HexagonInstrInfo &HII;
    MachineDominatorTree &MDT;
    MachineRegisterInfo &MRI;
  };
}


bool DeadCodeElimination::isDead(unsigned R) const {
  for (auto I = MRI.use_begin(R), E = MRI.use_end(); I != E; ++I) {
    MachineInstr *UseI = I->getParent();
    if (UseI->isDebugValue())
      continue;
    if (UseI->isPHI()) {
      assert(!UseI->getOperand(0).getSubReg());
      unsigned DR = UseI->getOperand(0).getReg();
      if (DR == R)
        continue;
    }
    return false;
  }
  return true;
}


bool DeadCodeElimination::runOnNode(MachineDomTreeNode *N) {
  bool Changed = false;
  typedef GraphTraits<MachineDomTreeNode*> GTN;
  for (auto I = GTN::child_begin(N), E = GTN::child_end(N); I != E; ++I)
    Changed |= runOnNode(*I);

  MachineBasicBlock *B = N->getBlock();
  std::vector<MachineInstr*> Instrs;
  for (auto I = B->rbegin(), E = B->rend(); I != E; ++I)
    Instrs.push_back(&*I);

  for (auto MI : Instrs) {
    unsigned Opc = MI->getOpcode();
    // Do not touch lifetime markers. This is why the target-independent DCE
    // cannot be used.
    if (Opc == TargetOpcode::LIFETIME_START ||
        Opc == TargetOpcode::LIFETIME_END)
      continue;
    bool Store = false;
    if (MI->isInlineAsm())
      continue;
    // Delete PHIs if possible.
    if (!MI->isPHI() && !MI->isSafeToMove(nullptr, Store))
      continue;

    bool AllDead = true;
    SmallVector<unsigned,2> Regs;
    for (auto &Op : MI->operands()) {
      if (!Op.isReg() || !Op.isDef())
        continue;
      unsigned R = Op.getReg();
      if (!TargetRegisterInfo::isVirtualRegister(R) || !isDead(R)) {
        AllDead = false;
        break;
      }
      Regs.push_back(R);
    }
    if (!AllDead)
      continue;

    B->erase(MI);
    for (unsigned i = 0, n = Regs.size(); i != n; ++i)
      MRI.markUsesInDebugValueAsUndef(Regs[i]);
    Changed = true;
  }

  return Changed;
}


//
// Eliminate redundant instructions
//
// This transformation will identify instructions where the output register
// is the same as one of its input registers. This only works on instructions
// that define a single register (unlike post-increment loads, for example).
// The equality check is actually more detailed: the code calculates which
// bits of the output are used, and only compares these bits with the input
// registers.
// If the output matches an input, the instruction is replaced with COPY.
// The copies will be removed by another transformation.
namespace {
  class RedundantInstrElimination : public Transformation {
  public:
    RedundantInstrElimination(BitTracker &bt, const HexagonInstrInfo &hii,
          MachineRegisterInfo &mri)
        : Transformation(true), HII(hii), MRI(mri), BT(bt) {}
    bool processBlock(MachineBasicBlock &B, const RegisterSet &AVs) override;
  private:
    bool isLossyShiftLeft(const MachineInstr &MI, unsigned OpN,
          unsigned &LostB, unsigned &LostE);
    bool isLossyShiftRight(const MachineInstr &MI, unsigned OpN,
          unsigned &LostB, unsigned &LostE);
    bool computeUsedBits(unsigned Reg, BitVector &Bits);
    bool computeUsedBits(const MachineInstr &MI, unsigned OpN, BitVector &Bits,
          uint16_t Begin);
    bool usedBitsEqual(BitTracker::RegisterRef RD, BitTracker::RegisterRef RS);

    const HexagonInstrInfo &HII;
    MachineRegisterInfo &MRI;
    BitTracker &BT;
  };
}


// Check if the instruction is a lossy shift left, where the input being
// shifted is the operand OpN of MI. If true, [LostB, LostE) is the range
// of bit indices that are lost.
bool RedundantInstrElimination::isLossyShiftLeft(const MachineInstr &MI,
      unsigned OpN, unsigned &LostB, unsigned &LostE) {
  using namespace Hexagon;
  unsigned Opc = MI.getOpcode();
  unsigned ImN, RegN, Width;
  switch (Opc) {
    case S2_asl_i_p:
      ImN = 2;
      RegN = 1;
      Width = 64;
      break;
    case S2_asl_i_p_acc:
    case S2_asl_i_p_and:
    case S2_asl_i_p_nac:
    case S2_asl_i_p_or:
    case S2_asl_i_p_xacc:
      ImN = 3;
      RegN = 2;
      Width = 64;
      break;
    case S2_asl_i_r:
      ImN = 2;
      RegN = 1;
      Width = 32;
      break;
    case S2_addasl_rrri:
    case S4_andi_asl_ri:
    case S4_ori_asl_ri:
    case S4_addi_asl_ri:
    case S4_subi_asl_ri:
    case S2_asl_i_r_acc:
    case S2_asl_i_r_and:
    case S2_asl_i_r_nac:
    case S2_asl_i_r_or:
    case S2_asl_i_r_sat:
    case S2_asl_i_r_xacc:
      ImN = 3;
      RegN = 2;
      Width = 32;
      break;
    default:
      return false;
  }

  if (RegN != OpN)
    return false;

  assert(MI.getOperand(ImN).isImm());
  unsigned S = MI.getOperand(ImN).getImm();
  if (S == 0)
    return false;
  LostB = Width-S;
  LostE = Width;
  return true;
}


// Check if the instruction is a lossy shift right, where the input being
// shifted is the operand OpN of MI. If true, [LostB, LostE) is the range
// of bit indices that are lost.
bool RedundantInstrElimination::isLossyShiftRight(const MachineInstr &MI,
      unsigned OpN, unsigned &LostB, unsigned &LostE) {
  using namespace Hexagon;
  unsigned Opc = MI.getOpcode();
  unsigned ImN, RegN;
  switch (Opc) {
    case S2_asr_i_p:
    case S2_lsr_i_p:
      ImN = 2;
      RegN = 1;
      break;
    case S2_asr_i_p_acc:
    case S2_asr_i_p_and:
    case S2_asr_i_p_nac:
    case S2_asr_i_p_or:
    case S2_lsr_i_p_acc:
    case S2_lsr_i_p_and:
    case S2_lsr_i_p_nac:
    case S2_lsr_i_p_or:
    case S2_lsr_i_p_xacc:
      ImN = 3;
      RegN = 2;
      break;
    case S2_asr_i_r:
    case S2_lsr_i_r:
      ImN = 2;
      RegN = 1;
      break;
    case S4_andi_lsr_ri:
    case S4_ori_lsr_ri:
    case S4_addi_lsr_ri:
    case S4_subi_lsr_ri:
    case S2_asr_i_r_acc:
    case S2_asr_i_r_and:
    case S2_asr_i_r_nac:
    case S2_asr_i_r_or:
    case S2_lsr_i_r_acc:
    case S2_lsr_i_r_and:
    case S2_lsr_i_r_nac:
    case S2_lsr_i_r_or:
    case S2_lsr_i_r_xacc:
      ImN = 3;
      RegN = 2;
      break;

    default:
      return false;
  }

  if (RegN != OpN)
    return false;

  assert(MI.getOperand(ImN).isImm());
  unsigned S = MI.getOperand(ImN).getImm();
  LostB = 0;
  LostE = S;
  return true;
}


// Calculate the bit vector that corresponds to the used bits of register Reg.
// The vector Bits has the same size, as the size of Reg in bits. If the cal-
// culation fails (i.e. the used bits are unknown), it returns false. Other-
// wise, it returns true and sets the corresponding bits in Bits.
bool RedundantInstrElimination::computeUsedBits(unsigned Reg, BitVector &Bits) {
  BitVector Used(Bits.size());
  RegisterSet Visited;
  std::vector<unsigned> Pending;
  Pending.push_back(Reg);

  for (unsigned i = 0; i < Pending.size(); ++i) {
    unsigned R = Pending[i];
    if (Visited.has(R))
      continue;
    Visited.insert(R);
    for (auto I = MRI.use_begin(R), E = MRI.use_end(); I != E; ++I) {
      BitTracker::RegisterRef UR = *I;
      unsigned B, W;
      if (!HBS::getSubregMask(UR, B, W, MRI))
        return false;
      MachineInstr &UseI = *I->getParent();
      if (UseI.isPHI() || UseI.isCopy()) {
        unsigned DefR = UseI.getOperand(0).getReg();
        if (!TargetRegisterInfo::isVirtualRegister(DefR))
          return false;
        Pending.push_back(DefR);
      } else {
        if (!computeUsedBits(UseI, I.getOperandNo(), Used, B))
          return false;
      }
    }
  }
  Bits |= Used;
  return true;
}


// Calculate the bits used by instruction MI in a register in operand OpN.
// Return true/false if the calculation succeeds/fails. If is succeeds, set
// used bits in Bits. This function does not reset any bits in Bits, so
// subsequent calls over different instructions will result in the union
// of the used bits in all these instructions.
// The register in question may be used with a sub-register, whereas Bits
// holds the bits for the entire register. To keep track of that, the
// argument Begin indicates where in Bits is the lowest-significant bit
// of the register used in operand OpN. For example, in instruction:
//   vreg1 = S2_lsr_i_r vreg2:subreg_hireg, 10
// the operand 1 is a 32-bit register, which happens to be a subregister
// of the 64-bit register vreg2, and that subregister starts at position 32.
// In this case Begin=32, since Bits[32] would be the lowest-significant bit
// of vreg2:subreg_hireg.
bool RedundantInstrElimination::computeUsedBits(const MachineInstr &MI,
      unsigned OpN, BitVector &Bits, uint16_t Begin) {
  unsigned Opc = MI.getOpcode();
  BitVector T(Bits.size());
  bool GotBits = HBS::getUsedBits(Opc, OpN, T, Begin, HII);
  // Even if we don't have bits yet, we could still provide some information
  // if the instruction is a lossy shift: the lost bits will be marked as
  // not used.
  unsigned LB, LE;
  if (isLossyShiftLeft(MI, OpN, LB, LE) || isLossyShiftRight(MI, OpN, LB, LE)) {
    assert(MI.getOperand(OpN).isReg());
    BitTracker::RegisterRef RR = MI.getOperand(OpN);
    const TargetRegisterClass *RC = HBS::getFinalVRegClass(RR, MRI);
    uint16_t Width = RC->getSize()*8;

    if (!GotBits)
      T.set(Begin, Begin+Width);
    assert(LB <= LE && LB < Width && LE <= Width);
    T.reset(Begin+LB, Begin+LE);
    GotBits = true;
  }
  if (GotBits)
    Bits |= T;
  return GotBits;
}


// Calculates the used bits in RD ("defined register"), and checks if these
// bits in RS ("used register") and RD are identical.
bool RedundantInstrElimination::usedBitsEqual(BitTracker::RegisterRef RD,
      BitTracker::RegisterRef RS) {
  const BitTracker::RegisterCell &DC = BT.lookup(RD.Reg);
  const BitTracker::RegisterCell &SC = BT.lookup(RS.Reg);

  unsigned DB, DW;
  if (!HBS::getSubregMask(RD, DB, DW, MRI))
    return false;
  unsigned SB, SW;
  if (!HBS::getSubregMask(RS, SB, SW, MRI))
    return false;
  if (SW != DW)
    return false;

  BitVector Used(DC.width());
  if (!computeUsedBits(RD.Reg, Used))
    return false;

  for (unsigned i = 0; i != DW; ++i)
    if (Used[i+DB] && DC[DB+i] != SC[SB+i])
      return false;
  return true;
}


bool RedundantInstrElimination::processBlock(MachineBasicBlock &B,
      const RegisterSet&) {
  bool Changed = false;

  for (auto I = B.begin(), E = B.end(), NextI = I; I != E; ++I) {
    NextI = std::next(I);
    MachineInstr *MI = &*I;

    if (MI->getOpcode() == TargetOpcode::COPY)
      continue;
    if (MI->hasUnmodeledSideEffects() || MI->isInlineAsm())
      continue;
    unsigned NumD = MI->getDesc().getNumDefs();
    if (NumD != 1)
      continue;

    BitTracker::RegisterRef RD = MI->getOperand(0);
    if (!BT.has(RD.Reg))
      continue;
    const BitTracker::RegisterCell &DC = BT.lookup(RD.Reg);
    auto At = MI->isPHI() ? B.getFirstNonPHI()
                          : MachineBasicBlock::iterator(MI);

    // Find a source operand that is equal to the result.
    for (auto &Op : MI->uses()) {
      if (!Op.isReg())
        continue;
      BitTracker::RegisterRef RS = Op;
      if (!BT.has(RS.Reg))
        continue;
      if (!HBS::isTransparentCopy(RD, RS, MRI))
        continue;

      unsigned BN, BW;
      if (!HBS::getSubregMask(RS, BN, BW, MRI))
        continue;

      const BitTracker::RegisterCell &SC = BT.lookup(RS.Reg);
      if (!usedBitsEqual(RD, RS) && !HBS::isEqual(DC, 0, SC, BN, BW))
        continue;

      // If found, replace the instruction with a COPY.
      const DebugLoc &DL = MI->getDebugLoc();
      const TargetRegisterClass *FRC = HBS::getFinalVRegClass(RD, MRI);
      unsigned NewR = MRI.createVirtualRegister(FRC);
      BuildMI(B, At, DL, HII.get(TargetOpcode::COPY), NewR)
          .addReg(RS.Reg, 0, RS.Sub);
      HBS::replaceSubWithSub(RD.Reg, RD.Sub, NewR, 0, MRI);
      BT.put(BitTracker::RegisterRef(NewR), SC);
      Changed = true;
      break;
    }
  }

  return Changed;
}


//
// Const generation
//
// Recognize instructions that produce constant values known at compile-time.
// Replace them with register definitions that load these constants directly.
namespace {
  class ConstGeneration : public Transformation {
  public:
    ConstGeneration(BitTracker &bt, const HexagonInstrInfo &hii,
        MachineRegisterInfo &mri)
      : Transformation(true), HII(hii), MRI(mri), BT(bt) {}
    bool processBlock(MachineBasicBlock &B, const RegisterSet &AVs) override;
  private:
    bool isTfrConst(const MachineInstr *MI) const;
    bool isConst(unsigned R, int64_t &V) const;
    unsigned genTfrConst(const TargetRegisterClass *RC, int64_t C,
        MachineBasicBlock &B, MachineBasicBlock::iterator At, DebugLoc &DL);

    const HexagonInstrInfo &HII;
    MachineRegisterInfo &MRI;
    BitTracker &BT;
  };
}

bool ConstGeneration::isConst(unsigned R, int64_t &C) const {
  if (!BT.has(R))
    return false;
  const BitTracker::RegisterCell &RC = BT.lookup(R);
  int64_t T = 0;
  for (unsigned i = RC.width(); i > 0; --i) {
    const BitTracker::BitValue &V = RC[i-1];
    T <<= 1;
    if (V.is(1))
      T |= 1;
    else if (!V.is(0))
      return false;
  }
  C = T;
  return true;
}


bool ConstGeneration::isTfrConst(const MachineInstr *MI) const {
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
    case Hexagon::A2_combineii:
    case Hexagon::A4_combineii:
    case Hexagon::A2_tfrsi:
    case Hexagon::A2_tfrpi:
    case Hexagon::TFR_PdTrue:
    case Hexagon::TFR_PdFalse:
    case Hexagon::CONST32_Int_Real:
    case Hexagon::CONST64_Int_Real:
      return true;
  }
  return false;
}


// Generate a transfer-immediate instruction that is appropriate for the
// register class and the actual value being transferred.
unsigned ConstGeneration::genTfrConst(const TargetRegisterClass *RC, int64_t C,
      MachineBasicBlock &B, MachineBasicBlock::iterator At, DebugLoc &DL) {
  unsigned Reg = MRI.createVirtualRegister(RC);
  if (RC == &Hexagon::IntRegsRegClass) {
    BuildMI(B, At, DL, HII.get(Hexagon::A2_tfrsi), Reg)
        .addImm(int32_t(C));
    return Reg;
  }

  if (RC == &Hexagon::DoubleRegsRegClass) {
    if (isInt<8>(C)) {
      BuildMI(B, At, DL, HII.get(Hexagon::A2_tfrpi), Reg)
          .addImm(C);
      return Reg;
    }

    unsigned Lo = Lo_32(C), Hi = Hi_32(C);
    if (isInt<8>(Lo) || isInt<8>(Hi)) {
      unsigned Opc = isInt<8>(Lo) ? Hexagon::A2_combineii
                                  : Hexagon::A4_combineii;
      BuildMI(B, At, DL, HII.get(Opc), Reg)
          .addImm(int32_t(Hi))
          .addImm(int32_t(Lo));
      return Reg;
    }

    BuildMI(B, At, DL, HII.get(Hexagon::CONST64_Int_Real), Reg)
        .addImm(C);
    return Reg;
  }

  if (RC == &Hexagon::PredRegsRegClass) {
    unsigned Opc;
    if (C == 0)
      Opc = Hexagon::TFR_PdFalse;
    else if ((C & 0xFF) == 0xFF)
      Opc = Hexagon::TFR_PdTrue;
    else
      return 0;
    BuildMI(B, At, DL, HII.get(Opc), Reg);
    return Reg;
  }

  return 0;
}


bool ConstGeneration::processBlock(MachineBasicBlock &B, const RegisterSet&) {
  bool Changed = false;
  RegisterSet Defs;

  for (auto I = B.begin(), E = B.end(); I != E; ++I) {
    if (isTfrConst(I))
      continue;
    Defs.clear();
    HBS::getInstrDefs(*I, Defs);
    if (Defs.count() != 1)
      continue;
    unsigned DR = Defs.find_first();
    if (!TargetRegisterInfo::isVirtualRegister(DR))
      continue;
    int64_t C;
    if (isConst(DR, C)) {
      DebugLoc DL = I->getDebugLoc();
      auto At = I->isPHI() ? B.getFirstNonPHI() : I;
      unsigned ImmReg = genTfrConst(MRI.getRegClass(DR), C, B, At, DL);
      if (ImmReg) {
        HBS::replaceReg(DR, ImmReg, MRI);
        BT.put(ImmReg, BT.lookup(DR));
        Changed = true;
      }
    }
  }
  return Changed;
}


//
// Copy generation
//
// Identify pairs of available registers which hold identical values.
// In such cases, only one of them needs to be calculated, the other one
// will be defined as a copy of the first.
//
// Copy propagation
//
// Eliminate register copies RD = RS, by replacing the uses of RD with
// with uses of RS.
namespace {
  class CopyGeneration : public Transformation {
  public:
    CopyGeneration(BitTracker &bt, const HexagonInstrInfo &hii,
        MachineRegisterInfo &mri)
      : Transformation(true), HII(hii), MRI(mri), BT(bt) {}
    bool processBlock(MachineBasicBlock &B, const RegisterSet &AVs) override;
  private:
    bool findMatch(const BitTracker::RegisterRef &Inp,
        BitTracker::RegisterRef &Out, const RegisterSet &AVs);

    const HexagonInstrInfo &HII;
    MachineRegisterInfo &MRI;
    BitTracker &BT;
  };

  class CopyPropagation : public Transformation {
  public:
    CopyPropagation(const HexagonRegisterInfo &hri, MachineRegisterInfo &mri)
        : Transformation(false), MRI(mri) {}
    bool processBlock(MachineBasicBlock &B, const RegisterSet &AVs) override;
    static bool isCopyReg(unsigned Opc);
  private:
    bool propagateRegCopy(MachineInstr &MI);

    MachineRegisterInfo &MRI;
  };

}


/// Check if there is a register in AVs that is identical to Inp. If so,
/// set Out to the found register. The output may be a pair Reg:Sub.
bool CopyGeneration::findMatch(const BitTracker::RegisterRef &Inp,
      BitTracker::RegisterRef &Out, const RegisterSet &AVs) {
  if (!BT.has(Inp.Reg))
    return false;
  const BitTracker::RegisterCell &InpRC = BT.lookup(Inp.Reg);
  unsigned B, W;
  if (!HBS::getSubregMask(Inp, B, W, MRI))
    return false;

  for (unsigned R = AVs.find_first(); R; R = AVs.find_next(R)) {
    if (!BT.has(R) || !HBS::isTransparentCopy(R, Inp, MRI))
      continue;
    const BitTracker::RegisterCell &RC = BT.lookup(R);
    unsigned RW = RC.width();
    if (W == RW) {
      if (MRI.getRegClass(Inp.Reg) != MRI.getRegClass(R))
        continue;
      if (!HBS::isEqual(InpRC, B, RC, 0, W))
        continue;
      Out.Reg = R;
      Out.Sub = 0;
      return true;
    }
    // Check if there is a super-register, whose part (with a subregister)
    // is equal to the input.
    // Only do double registers for now.
    if (W*2 != RW)
      continue;
    if (MRI.getRegClass(R) != &Hexagon::DoubleRegsRegClass)
      continue;

    if (HBS::isEqual(InpRC, B, RC, 0, W))
      Out.Sub = Hexagon::subreg_loreg;
    else if (HBS::isEqual(InpRC, B, RC, W, W))
      Out.Sub = Hexagon::subreg_hireg;
    else
      continue;
    Out.Reg = R;
    return true;
  }
  return false;
}


bool CopyGeneration::processBlock(MachineBasicBlock &B,
      const RegisterSet &AVs) {
  RegisterSet AVB(AVs);
  bool Changed = false;
  RegisterSet Defs;

  for (auto I = B.begin(), E = B.end(), NextI = I; I != E;
       ++I, AVB.insert(Defs)) {
    NextI = std::next(I);
    Defs.clear();
    HBS::getInstrDefs(*I, Defs);

    unsigned Opc = I->getOpcode();
    if (CopyPropagation::isCopyReg(Opc))
      continue;

    for (unsigned R = Defs.find_first(); R; R = Defs.find_next(R)) {
      BitTracker::RegisterRef MR;
      if (!findMatch(R, MR, AVB))
        continue;
      DebugLoc DL = I->getDebugLoc();
      auto *FRC = HBS::getFinalVRegClass(MR, MRI);
      unsigned NewR = MRI.createVirtualRegister(FRC);
      auto At = I->isPHI() ? B.getFirstNonPHI() : I;
      BuildMI(B, At, DL, HII.get(TargetOpcode::COPY), NewR)
        .addReg(MR.Reg, 0, MR.Sub);
      BT.put(BitTracker::RegisterRef(NewR), BT.get(MR));
    }
  }

  return Changed;
}


bool CopyPropagation::isCopyReg(unsigned Opc) {
  switch (Opc) {
    case TargetOpcode::COPY:
    case TargetOpcode::REG_SEQUENCE:
    case Hexagon::A2_tfr:
    case Hexagon::A2_tfrp:
    case Hexagon::A2_combinew:
    case Hexagon::A4_combineir:
    case Hexagon::A4_combineri:
      return true;
    default:
      break;
  }
  return false;
}


bool CopyPropagation::propagateRegCopy(MachineInstr &MI) {
  bool Changed = false;
  unsigned Opc = MI.getOpcode();
  BitTracker::RegisterRef RD = MI.getOperand(0);
  assert(MI.getOperand(0).getSubReg() == 0);

  switch (Opc) {
    case TargetOpcode::COPY:
    case Hexagon::A2_tfr:
    case Hexagon::A2_tfrp: {
      BitTracker::RegisterRef RS = MI.getOperand(1);
      if (!HBS::isTransparentCopy(RD, RS, MRI))
        break;
      if (RS.Sub != 0)
        Changed = HBS::replaceRegWithSub(RD.Reg, RS.Reg, RS.Sub, MRI);
      else
        Changed = HBS::replaceReg(RD.Reg, RS.Reg, MRI);
      break;
    }
    case TargetOpcode::REG_SEQUENCE: {
      BitTracker::RegisterRef SL, SH;
      if (HBS::parseRegSequence(MI, SL, SH)) {
        Changed = HBS::replaceSubWithSub(RD.Reg, Hexagon::subreg_loreg,
                                         SL.Reg, SL.Sub, MRI);
        Changed |= HBS::replaceSubWithSub(RD.Reg, Hexagon::subreg_hireg,
                                          SH.Reg, SH.Sub, MRI);
      }
      break;
    }
    case Hexagon::A2_combinew: {
      BitTracker::RegisterRef RH = MI.getOperand(1), RL = MI.getOperand(2);
      Changed = HBS::replaceSubWithSub(RD.Reg, Hexagon::subreg_loreg,
                                       RL.Reg, RL.Sub, MRI);
      Changed |= HBS::replaceSubWithSub(RD.Reg, Hexagon::subreg_hireg,
                                        RH.Reg, RH.Sub, MRI);
      break;
    }
    case Hexagon::A4_combineir:
    case Hexagon::A4_combineri: {
      unsigned SrcX = (Opc == Hexagon::A4_combineir) ? 2 : 1;
      unsigned Sub = (Opc == Hexagon::A4_combineir) ? Hexagon::subreg_loreg
                                                    : Hexagon::subreg_hireg;
      BitTracker::RegisterRef RS = MI.getOperand(SrcX);
      Changed = HBS::replaceSubWithSub(RD.Reg, Sub, RS.Reg, RS.Sub, MRI);
      break;
    }
  }
  return Changed;
}


bool CopyPropagation::processBlock(MachineBasicBlock &B, const RegisterSet&) {
  std::vector<MachineInstr*> Instrs;
  for (auto I = B.rbegin(), E = B.rend(); I != E; ++I)
    Instrs.push_back(&*I);

  bool Changed = false;
  for (auto I : Instrs) {
    unsigned Opc = I->getOpcode();
    if (!CopyPropagation::isCopyReg(Opc))
      continue;
    Changed |= propagateRegCopy(*I);
  }

  return Changed;
}


//
// Bit simplification
//
// Recognize patterns that can be simplified and replace them with the
// simpler forms.
// This is by no means complete
namespace {
  class BitSimplification : public Transformation {
  public:
    BitSimplification(BitTracker &bt, const HexagonInstrInfo &hii,
        MachineRegisterInfo &mri)
      : Transformation(true), HII(hii), MRI(mri), BT(bt) {}
    bool processBlock(MachineBasicBlock &B, const RegisterSet &AVs) override;
  private:
    struct RegHalf : public BitTracker::RegisterRef {
      bool Low;  // Low/High halfword.
    };

    bool matchHalf(unsigned SelfR, const BitTracker::RegisterCell &RC,
          unsigned B, RegHalf &RH);

    bool matchPackhl(unsigned SelfR, const BitTracker::RegisterCell &RC,
          BitTracker::RegisterRef &Rs, BitTracker::RegisterRef &Rt);
    unsigned getCombineOpcode(bool HLow, bool LLow);

    bool genStoreUpperHalf(MachineInstr *MI);
    bool genStoreImmediate(MachineInstr *MI);
    bool genPackhl(MachineInstr *MI, BitTracker::RegisterRef RD,
          const BitTracker::RegisterCell &RC);
    bool genExtractHalf(MachineInstr *MI, BitTracker::RegisterRef RD,
          const BitTracker::RegisterCell &RC);
    bool genCombineHalf(MachineInstr *MI, BitTracker::RegisterRef RD,
          const BitTracker::RegisterCell &RC);
    bool genExtractLow(MachineInstr *MI, BitTracker::RegisterRef RD,
          const BitTracker::RegisterCell &RC);
    bool simplifyTstbit(MachineInstr *MI, BitTracker::RegisterRef RD,
          const BitTracker::RegisterCell &RC);

    const HexagonInstrInfo &HII;
    MachineRegisterInfo &MRI;
    BitTracker &BT;
  };
}


// Check if the bits [B..B+16) in register cell RC form a valid halfword,
// i.e. [0..16), [16..32), etc. of some register. If so, return true and
// set the information about the found register in RH.
bool BitSimplification::matchHalf(unsigned SelfR,
      const BitTracker::RegisterCell &RC, unsigned B, RegHalf &RH) {
  // XXX This could be searching in the set of available registers, in case
  // the match is not exact.

  // Match 16-bit chunks, where the RC[B..B+15] references exactly one
  // register and all the bits B..B+15 match between RC and the register.
  // This is meant to match "v1[0-15]", where v1 = { [0]:0 [1-15]:v1... },
  // and RC = { [0]:0 [1-15]:v1[1-15]... }.
  bool Low = false;
  unsigned I = B;
  while (I < B+16 && RC[I].num())
    I++;
  if (I == B+16)
    return false;

  unsigned Reg = RC[I].RefI.Reg;
  unsigned P = RC[I].RefI.Pos;    // The RefI.Pos will be advanced by I-B.
  if (P < I-B)
    return false;
  unsigned Pos = P - (I-B);

  if (Reg == 0 || Reg == SelfR)    // Don't match "self".
    return false;
  if (!TargetRegisterInfo::isVirtualRegister(Reg))
    return false;
  if (!BT.has(Reg))
    return false;

  const BitTracker::RegisterCell &SC = BT.lookup(Reg);
  if (Pos+16 > SC.width())
    return false;

  for (unsigned i = 0; i < 16; ++i) {
    const BitTracker::BitValue &RV = RC[i+B];
    if (RV.Type == BitTracker::BitValue::Ref) {
      if (RV.RefI.Reg != Reg)
        return false;
      if (RV.RefI.Pos != i+Pos)
        return false;
      continue;
    }
    if (RC[i+B] != SC[i+Pos])
      return false;
  }

  unsigned Sub = 0;
  switch (Pos) {
    case 0:
      Sub = Hexagon::subreg_loreg;
      Low = true;
      break;
    case 16:
      Sub = Hexagon::subreg_loreg;
      Low = false;
      break;
    case 32:
      Sub = Hexagon::subreg_hireg;
      Low = true;
      break;
    case 48:
      Sub = Hexagon::subreg_hireg;
      Low = false;
      break;
    default:
      return false;
  }

  RH.Reg = Reg;
  RH.Sub = Sub;
  RH.Low = Low;
  // If the subregister is not valid with the register, set it to 0.
  if (!HBS::getFinalVRegClass(RH, MRI))
    RH.Sub = 0;

  return true;
}


// Check if RC matches the pattern of a S2_packhl. If so, return true and
// set the inputs Rs and Rt.
bool BitSimplification::matchPackhl(unsigned SelfR,
      const BitTracker::RegisterCell &RC, BitTracker::RegisterRef &Rs,
      BitTracker::RegisterRef &Rt) {
  RegHalf L1, H1, L2, H2;

  if (!matchHalf(SelfR, RC, 0, L2)  || !matchHalf(SelfR, RC, 16, L1))
    return false;
  if (!matchHalf(SelfR, RC, 32, H2) || !matchHalf(SelfR, RC, 48, H1))
    return false;

  // Rs = H1.L1, Rt = H2.L2
  if (H1.Reg != L1.Reg || H1.Sub != L1.Sub || H1.Low || !L1.Low)
    return false;
  if (H2.Reg != L2.Reg || H2.Sub != L2.Sub || H2.Low || !L2.Low)
    return false;

  Rs = H1;
  Rt = H2;
  return true;
}


unsigned BitSimplification::getCombineOpcode(bool HLow, bool LLow) {
  return HLow ? LLow ? Hexagon::A2_combine_ll
                     : Hexagon::A2_combine_lh
              : LLow ? Hexagon::A2_combine_hl
                     : Hexagon::A2_combine_hh;
}


// If MI stores the upper halfword of a register (potentially obtained via
// shifts or extracts), replace it with a storerf instruction. This could
// cause the "extraction" code to become dead.
bool BitSimplification::genStoreUpperHalf(MachineInstr *MI) {
  unsigned Opc = MI->getOpcode();
  if (Opc != Hexagon::S2_storerh_io)
    return false;

  MachineOperand &ValOp = MI->getOperand(2);
  BitTracker::RegisterRef RS = ValOp;
  if (!BT.has(RS.Reg))
    return false;
  const BitTracker::RegisterCell &RC = BT.lookup(RS.Reg);
  RegHalf H;
  if (!matchHalf(0, RC, 0, H))
    return false;
  if (H.Low)
    return false;
  MI->setDesc(HII.get(Hexagon::S2_storerf_io));
  ValOp.setReg(H.Reg);
  ValOp.setSubReg(H.Sub);
  return true;
}


// If MI stores a value known at compile-time, and the value is within a range
// that avoids using constant-extenders, replace it with a store-immediate.
bool BitSimplification::genStoreImmediate(MachineInstr *MI) {
  unsigned Opc = MI->getOpcode();
  unsigned Align = 0;
  switch (Opc) {
    case Hexagon::S2_storeri_io:
      Align++;
    case Hexagon::S2_storerh_io:
      Align++;
    case Hexagon::S2_storerb_io:
      break;
    default:
      return false;
  }

  // Avoid stores to frame-indices (due to an unknown offset).
  if (!MI->getOperand(0).isReg())
    return false;
  MachineOperand &OffOp = MI->getOperand(1);
  if (!OffOp.isImm())
    return false;

  int64_t Off = OffOp.getImm();
  // Offset is u6:a. Sadly, there is no isShiftedUInt(n,x).
  if (!isUIntN(6+Align, Off) || (Off & ((1<<Align)-1)))
    return false;
  // Source register:
  BitTracker::RegisterRef RS = MI->getOperand(2);
  if (!BT.has(RS.Reg))
    return false;
  const BitTracker::RegisterCell &RC = BT.lookup(RS.Reg);
  uint64_t U;
  if (!HBS::getConst(RC, 0, RC.width(), U))
    return false;

  // Only consider 8-bit values to avoid constant-extenders.
  int V;
  switch (Opc) {
    case Hexagon::S2_storerb_io:
      V = int8_t(U);
      break;
    case Hexagon::S2_storerh_io:
      V = int16_t(U);
      break;
    case Hexagon::S2_storeri_io:
      V = int32_t(U);
      break;
  }
  if (!isInt<8>(V))
    return false;

  MI->RemoveOperand(2);
  switch (Opc) {
    case Hexagon::S2_storerb_io:
      MI->setDesc(HII.get(Hexagon::S4_storeirb_io));
      break;
    case Hexagon::S2_storerh_io:
      MI->setDesc(HII.get(Hexagon::S4_storeirh_io));
      break;
    case Hexagon::S2_storeri_io:
      MI->setDesc(HII.get(Hexagon::S4_storeiri_io));
      break;
  }
  MI->addOperand(MachineOperand::CreateImm(V));
  return true;
}


// If MI is equivalent o S2_packhl, generate the S2_packhl. MI could be the
// last instruction in a sequence that results in something equivalent to
// the pack-halfwords. The intent is to cause the entire sequence to become
// dead.
bool BitSimplification::genPackhl(MachineInstr *MI,
      BitTracker::RegisterRef RD, const BitTracker::RegisterCell &RC) {
  unsigned Opc = MI->getOpcode();
  if (Opc == Hexagon::S2_packhl)
    return false;
  BitTracker::RegisterRef Rs, Rt;
  if (!matchPackhl(RD.Reg, RC, Rs, Rt))
    return false;

  MachineBasicBlock &B = *MI->getParent();
  unsigned NewR = MRI.createVirtualRegister(&Hexagon::DoubleRegsRegClass);
  DebugLoc DL = MI->getDebugLoc();
  auto At = MI->isPHI() ? B.getFirstNonPHI()
                        : MachineBasicBlock::iterator(MI);
  BuildMI(B, At, DL, HII.get(Hexagon::S2_packhl), NewR)
      .addReg(Rs.Reg, 0, Rs.Sub)
      .addReg(Rt.Reg, 0, Rt.Sub);
  HBS::replaceSubWithSub(RD.Reg, RD.Sub, NewR, 0, MRI);
  BT.put(BitTracker::RegisterRef(NewR), RC);
  return true;
}


// If MI produces halfword of the input in the low half of the output,
// replace it with zero-extend or extractu.
bool BitSimplification::genExtractHalf(MachineInstr *MI,
      BitTracker::RegisterRef RD, const BitTracker::RegisterCell &RC) {
  RegHalf L;
  // Check for halfword in low 16 bits, zeros elsewhere.
  if (!matchHalf(RD.Reg, RC, 0, L) || !HBS::isZero(RC, 16, 16))
    return false;

  unsigned Opc = MI->getOpcode();
  MachineBasicBlock &B = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();

  // Prefer zxth, since zxth can go in any slot, while extractu only in
  // slots 2 and 3.
  unsigned NewR = 0;
  auto At = MI->isPHI() ? B.getFirstNonPHI()
                        : MachineBasicBlock::iterator(MI);
  if (L.Low && Opc != Hexagon::A2_zxth) {
    NewR = MRI.createVirtualRegister(&Hexagon::IntRegsRegClass);
    BuildMI(B, At, DL, HII.get(Hexagon::A2_zxth), NewR)
        .addReg(L.Reg, 0, L.Sub);
  } else if (!L.Low && Opc != Hexagon::S2_lsr_i_r) {
    NewR = MRI.createVirtualRegister(&Hexagon::IntRegsRegClass);
    BuildMI(B, MI, DL, HII.get(Hexagon::S2_lsr_i_r), NewR)
        .addReg(L.Reg, 0, L.Sub)
        .addImm(16);
  }
  if (NewR == 0)
    return false;
  HBS::replaceSubWithSub(RD.Reg, RD.Sub, NewR, 0, MRI);
  BT.put(BitTracker::RegisterRef(NewR), RC);
  return true;
}


// If MI is equivalent to a combine(.L/.H, .L/.H) replace with with the
// combine.
bool BitSimplification::genCombineHalf(MachineInstr *MI,
      BitTracker::RegisterRef RD, const BitTracker::RegisterCell &RC) {
  RegHalf L, H;
  // Check for combine h/l
  if (!matchHalf(RD.Reg, RC, 0, L) || !matchHalf(RD.Reg, RC, 16, H))
    return false;
  // Do nothing if this is just a reg copy.
  if (L.Reg == H.Reg && L.Sub == H.Sub && !H.Low && L.Low)
    return false;

  unsigned Opc = MI->getOpcode();
  unsigned COpc = getCombineOpcode(H.Low, L.Low);
  if (COpc == Opc)
    return false;

  MachineBasicBlock &B = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();
  unsigned NewR = MRI.createVirtualRegister(&Hexagon::IntRegsRegClass);
  auto At = MI->isPHI() ? B.getFirstNonPHI()
                        : MachineBasicBlock::iterator(MI);
  BuildMI(B, At, DL, HII.get(COpc), NewR)
      .addReg(H.Reg, 0, H.Sub)
      .addReg(L.Reg, 0, L.Sub);
  HBS::replaceSubWithSub(RD.Reg, RD.Sub, NewR, 0, MRI);
  BT.put(BitTracker::RegisterRef(NewR), RC);
  return true;
}


// If MI resets high bits of a register and keeps the lower ones, replace it
// with zero-extend byte/half, and-immediate, or extractu, as appropriate.
bool BitSimplification::genExtractLow(MachineInstr *MI,
      BitTracker::RegisterRef RD, const BitTracker::RegisterCell &RC) {
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
    case Hexagon::A2_zxtb:
    case Hexagon::A2_zxth:
    case Hexagon::S2_extractu:
      return false;
  }
  if (Opc == Hexagon::A2_andir && MI->getOperand(2).isImm()) {
    int32_t Imm = MI->getOperand(2).getImm();
    if (isInt<10>(Imm))
      return false;
  }

  if (MI->hasUnmodeledSideEffects() || MI->isInlineAsm())
    return false;
  unsigned W = RC.width();
  while (W > 0 && RC[W-1].is(0))
    W--;
  if (W == 0 || W == RC.width())
    return false;
  unsigned NewOpc = (W == 8)  ? Hexagon::A2_zxtb
                  : (W == 16) ? Hexagon::A2_zxth
                  : (W < 10)  ? Hexagon::A2_andir
                  : Hexagon::S2_extractu;
  MachineBasicBlock &B = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();

  for (auto &Op : MI->uses()) {
    if (!Op.isReg())
      continue;
    BitTracker::RegisterRef RS = Op;
    if (!BT.has(RS.Reg))
      continue;
    const BitTracker::RegisterCell &SC = BT.lookup(RS.Reg);
    unsigned BN, BW;
    if (!HBS::getSubregMask(RS, BN, BW, MRI))
      continue;
    if (BW < W || !HBS::isEqual(RC, 0, SC, BN, W))
      continue;

    unsigned NewR = MRI.createVirtualRegister(&Hexagon::IntRegsRegClass);
    auto At = MI->isPHI() ? B.getFirstNonPHI()
                          : MachineBasicBlock::iterator(MI);
    auto MIB = BuildMI(B, At, DL, HII.get(NewOpc), NewR)
                  .addReg(RS.Reg, 0, RS.Sub);
    if (NewOpc == Hexagon::A2_andir)
      MIB.addImm((1 << W) - 1);
    else if (NewOpc == Hexagon::S2_extractu)
      MIB.addImm(W).addImm(0);
    HBS::replaceSubWithSub(RD.Reg, RD.Sub, NewR, 0, MRI);
    BT.put(BitTracker::RegisterRef(NewR), RC);
    return true;
  }
  return false;
}


// Check for tstbit simplification opportunity, where the bit being checked
// can be tracked back to another register. For example:
//   vreg2 = S2_lsr_i_r  vreg1, 5
//   vreg3 = S2_tstbit_i vreg2, 0
// =>
//   vreg3 = S2_tstbit_i vreg1, 5
bool BitSimplification::simplifyTstbit(MachineInstr *MI,
      BitTracker::RegisterRef RD, const BitTracker::RegisterCell &RC) {
  unsigned Opc = MI->getOpcode();
  if (Opc != Hexagon::S2_tstbit_i)
    return false;

  unsigned BN = MI->getOperand(2).getImm();
  BitTracker::RegisterRef RS = MI->getOperand(1);
  unsigned F, W;
  DebugLoc DL = MI->getDebugLoc();
  if (!BT.has(RS.Reg) || !HBS::getSubregMask(RS, F, W, MRI))
    return false;
  MachineBasicBlock &B = *MI->getParent();
  auto At = MI->isPHI() ? B.getFirstNonPHI()
                        : MachineBasicBlock::iterator(MI);

  const BitTracker::RegisterCell &SC = BT.lookup(RS.Reg);
  const BitTracker::BitValue &V = SC[F+BN];
  if (V.Type == BitTracker::BitValue::Ref && V.RefI.Reg != RS.Reg) {
    const TargetRegisterClass *TC = MRI.getRegClass(V.RefI.Reg);
    // Need to map V.RefI.Reg to a 32-bit register, i.e. if it is
    // a double register, need to use a subregister and adjust bit
    // number.
    unsigned P = UINT_MAX;
    BitTracker::RegisterRef RR(V.RefI.Reg, 0);
    if (TC == &Hexagon::DoubleRegsRegClass) {
      P = V.RefI.Pos;
      RR.Sub = Hexagon::subreg_loreg;
      if (P >= 32) {
        P -= 32;
        RR.Sub = Hexagon::subreg_hireg;
      }
    } else if (TC == &Hexagon::IntRegsRegClass) {
      P = V.RefI.Pos;
    }
    if (P != UINT_MAX) {
      unsigned NewR = MRI.createVirtualRegister(&Hexagon::PredRegsRegClass);
      BuildMI(B, At, DL, HII.get(Hexagon::S2_tstbit_i), NewR)
          .addReg(RR.Reg, 0, RR.Sub)
          .addImm(P);
      HBS::replaceReg(RD.Reg, NewR, MRI);
      BT.put(NewR, RC);
      return true;
    }
  } else if (V.is(0) || V.is(1)) {
    unsigned NewR = MRI.createVirtualRegister(&Hexagon::PredRegsRegClass);
    unsigned NewOpc = V.is(0) ? Hexagon::TFR_PdFalse : Hexagon::TFR_PdTrue;
    BuildMI(B, At, DL, HII.get(NewOpc), NewR);
    HBS::replaceReg(RD.Reg, NewR, MRI);
    return true;
  }

  return false;
}


bool BitSimplification::processBlock(MachineBasicBlock &B,
      const RegisterSet &AVs) {
  bool Changed = false;
  RegisterSet AVB = AVs;
  RegisterSet Defs;

  for (auto I = B.begin(), E = B.end(); I != E; ++I, AVB.insert(Defs)) {
    MachineInstr *MI = &*I;
    Defs.clear();
    HBS::getInstrDefs(*MI, Defs);

    unsigned Opc = MI->getOpcode();
    if (Opc == TargetOpcode::COPY || Opc == TargetOpcode::REG_SEQUENCE)
      continue;

    if (MI->mayStore()) {
      bool T = genStoreUpperHalf(MI);
      T = T || genStoreImmediate(MI);
      Changed |= T;
      continue;
    }

    if (Defs.count() != 1)
      continue;
    const MachineOperand &Op0 = MI->getOperand(0);
    if (!Op0.isReg() || !Op0.isDef())
      continue;
    BitTracker::RegisterRef RD = Op0;
    if (!BT.has(RD.Reg))
      continue;
    const TargetRegisterClass *FRC = HBS::getFinalVRegClass(RD, MRI);
    const BitTracker::RegisterCell &RC = BT.lookup(RD.Reg);

    if (FRC->getID() == Hexagon::DoubleRegsRegClassID) {
      bool T = genPackhl(MI, RD, RC);
      Changed |= T;
      continue;
    }

    if (FRC->getID() == Hexagon::IntRegsRegClassID) {
      bool T = genExtractHalf(MI, RD, RC);
      T = T || genCombineHalf(MI, RD, RC);
      T = T || genExtractLow(MI, RD, RC);
      Changed |= T;
      continue;
    }

    if (FRC->getID() == Hexagon::PredRegsRegClassID) {
      bool T = simplifyTstbit(MI, RD, RC);
      Changed |= T;
      continue;
    }
  }
  return Changed;
}


bool HexagonBitSimplify::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  auto &HST = MF.getSubtarget<HexagonSubtarget>();
  auto &HRI = *HST.getRegisterInfo();
  auto &HII = *HST.getInstrInfo();

  MDT = &getAnalysis<MachineDominatorTree>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool Changed;

  Changed = DeadCodeElimination(MF, *MDT).run();

  const HexagonEvaluator HE(HRI, MRI, HII, MF);
  BitTracker BT(HE, MF);
  DEBUG(BT.trace(true));
  BT.run();

  MachineBasicBlock &Entry = MF.front();

  RegisterSet AIG;  // Available registers for IG.
  ConstGeneration ImmG(BT, HII, MRI);
  Changed |= visitBlock(Entry, ImmG, AIG);

  RegisterSet ARE;  // Available registers for RIE.
  RedundantInstrElimination RIE(BT, HII, MRI);
  Changed |= visitBlock(Entry, RIE, ARE);

  RegisterSet ACG;  // Available registers for CG.
  CopyGeneration CopyG(BT, HII, MRI);
  Changed |= visitBlock(Entry, CopyG, ACG);

  RegisterSet ACP;  // Available registers for CP.
  CopyPropagation CopyP(HRI, MRI);
  Changed |= visitBlock(Entry, CopyP, ACP);

  Changed = DeadCodeElimination(MF, *MDT).run() || Changed;

  BT.run();
  RegisterSet ABS;  // Available registers for BS.
  BitSimplification BitS(BT, HII, MRI);
  Changed |= visitBlock(Entry, BitS, ABS);

  Changed = DeadCodeElimination(MF, *MDT).run() || Changed;

  if (Changed) {
    for (auto &B : MF)
      for (auto &I : B)
        I.clearKillInfo();
    DeadCodeElimination(MF, *MDT).run();
  }
  return Changed;
}


// Recognize loops where the code at the end of the loop matches the code
// before the entry of the loop, and the matching code is such that is can
// be simplified. This pass relies on the bit simplification above and only
// prepares code in a way that can be handled by the bit simplifcation.
//
// This is the motivating testcase (and explanation):
//
// {
//   loop0(.LBB0_2, r1)      // %for.body.preheader
//   r5:4 = memd(r0++#8)
// }
// {
//   r3 = lsr(r4, #16)
//   r7:6 = combine(r5, r5)
// }
// {
//   r3 = insert(r5, #16, #16)
//   r7:6 = vlsrw(r7:6, #16)
// }
// .LBB0_2:
// {
//   memh(r2+#4) = r5
//   memh(r2+#6) = r6            # R6 is really R5.H
// }
// {
//   r2 = add(r2, #8)
//   memh(r2+#0) = r4
//   memh(r2+#2) = r3            # R3 is really R4.H
// }
// {
//   r5:4 = memd(r0++#8)
// }
// {                             # "Shuffling" code that sets up R3 and R6
//   r3 = lsr(r4, #16)           # so that their halves can be stored in the
//   r7:6 = combine(r5, r5)      # next iteration. This could be folded into
// }                             # the stores if the code was at the beginning
// {                             # of the loop iteration. Since the same code
//   r3 = insert(r5, #16, #16)   # precedes the loop, it can actually be moved
//   r7:6 = vlsrw(r7:6, #16)     # there.
// }:endloop0
//
//
// The outcome:
//
// {
//   loop0(.LBB0_2, r1)
//   r5:4 = memd(r0++#8)
// }
// .LBB0_2:
// {
//   memh(r2+#4) = r5
//   memh(r2+#6) = r5.h
// }
// {
//   r2 = add(r2, #8)
//   memh(r2+#0) = r4
//   memh(r2+#2) = r4.h
// }
// {
//   r5:4 = memd(r0++#8)
// }:endloop0

namespace llvm {
  FunctionPass *createHexagonLoopRescheduling();
  void initializeHexagonLoopReschedulingPass(PassRegistry&);
}

namespace {
  class HexagonLoopRescheduling : public MachineFunctionPass {
  public:
    static char ID;
    HexagonLoopRescheduling() : MachineFunctionPass(ID),
        HII(0), HRI(0), MRI(0), BTP(0) {
      initializeHexagonLoopReschedulingPass(*PassRegistry::getPassRegistry());
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

  private:
    const HexagonInstrInfo *HII;
    const HexagonRegisterInfo *HRI;
    MachineRegisterInfo *MRI;
    BitTracker *BTP;

    struct LoopCand {
      LoopCand(MachineBasicBlock *lb, MachineBasicBlock *pb,
            MachineBasicBlock *eb) : LB(lb), PB(pb), EB(eb) {}
      MachineBasicBlock *LB, *PB, *EB;
    };
    typedef std::vector<MachineInstr*> InstrList;
    struct InstrGroup {
      BitTracker::RegisterRef Inp, Out;
      InstrList Ins;
    };
    struct PhiInfo {
      PhiInfo(MachineInstr &P, MachineBasicBlock &B);
      unsigned DefR;
      BitTracker::RegisterRef LR, PR;
      MachineBasicBlock *LB, *PB;
    };

    static unsigned getDefReg(const MachineInstr *MI);
    bool isConst(unsigned Reg) const;
    bool isBitShuffle(const MachineInstr *MI, unsigned DefR) const;
    bool isStoreInput(const MachineInstr *MI, unsigned DefR) const;
    bool isShuffleOf(unsigned OutR, unsigned InpR) const;
    bool isSameShuffle(unsigned OutR1, unsigned InpR1, unsigned OutR2,
        unsigned &InpR2) const;
    void moveGroup(InstrGroup &G, MachineBasicBlock &LB, MachineBasicBlock &PB,
        MachineBasicBlock::iterator At, unsigned OldPhiR, unsigned NewPredR);
    bool processLoop(LoopCand &C);
  };
}

char HexagonLoopRescheduling::ID = 0;

INITIALIZE_PASS(HexagonLoopRescheduling, "hexagon-loop-resched",
  "Hexagon Loop Rescheduling", false, false)


HexagonLoopRescheduling::PhiInfo::PhiInfo(MachineInstr &P,
      MachineBasicBlock &B) {
  DefR = HexagonLoopRescheduling::getDefReg(&P);
  LB = &B;
  PB = nullptr;
  for (unsigned i = 1, n = P.getNumOperands(); i < n; i += 2) {
    const MachineOperand &OpB = P.getOperand(i+1);
    if (OpB.getMBB() == &B) {
      LR = P.getOperand(i);
      continue;
    }
    PB = OpB.getMBB();
    PR = P.getOperand(i);
  }
}


unsigned HexagonLoopRescheduling::getDefReg(const MachineInstr *MI) {
  RegisterSet Defs;
  HBS::getInstrDefs(*MI, Defs);
  if (Defs.count() != 1)
    return 0;
  return Defs.find_first();
}


bool HexagonLoopRescheduling::isConst(unsigned Reg) const {
  if (!BTP->has(Reg))
    return false;
  const BitTracker::RegisterCell &RC = BTP->lookup(Reg);
  for (unsigned i = 0, w = RC.width(); i < w; ++i) {
    const BitTracker::BitValue &V = RC[i];
    if (!V.is(0) && !V.is(1))
      return false;
  }
  return true;
}


bool HexagonLoopRescheduling::isBitShuffle(const MachineInstr *MI,
      unsigned DefR) const {
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
    case TargetOpcode::COPY:
    case Hexagon::S2_lsr_i_r:
    case Hexagon::S2_asr_i_r:
    case Hexagon::S2_asl_i_r:
    case Hexagon::S2_lsr_i_p:
    case Hexagon::S2_asr_i_p:
    case Hexagon::S2_asl_i_p:
    case Hexagon::S2_insert:
    case Hexagon::A2_or:
    case Hexagon::A2_orp:
    case Hexagon::A2_and:
    case Hexagon::A2_andp:
    case Hexagon::A2_combinew:
    case Hexagon::A4_combineri:
    case Hexagon::A4_combineir:
    case Hexagon::A2_combineii:
    case Hexagon::A4_combineii:
    case Hexagon::A2_combine_ll:
    case Hexagon::A2_combine_lh:
    case Hexagon::A2_combine_hl:
    case Hexagon::A2_combine_hh:
      return true;
  }
  return false;
}


bool HexagonLoopRescheduling::isStoreInput(const MachineInstr *MI,
      unsigned InpR) const {
  for (unsigned i = 0, n = MI->getNumOperands(); i < n; ++i) {
    const MachineOperand &Op = MI->getOperand(i);
    if (!Op.isReg())
      continue;
    if (Op.getReg() == InpR)
      return i == n-1;
  }
  return false;
}


bool HexagonLoopRescheduling::isShuffleOf(unsigned OutR, unsigned InpR) const {
  if (!BTP->has(OutR) || !BTP->has(InpR))
    return false;
  const BitTracker::RegisterCell &OutC = BTP->lookup(OutR);
  for (unsigned i = 0, w = OutC.width(); i < w; ++i) {
    const BitTracker::BitValue &V = OutC[i];
    if (V.Type != BitTracker::BitValue::Ref)
      continue;
    if (V.RefI.Reg != InpR)
      return false;
  }
  return true;
}


bool HexagonLoopRescheduling::isSameShuffle(unsigned OutR1, unsigned InpR1,
      unsigned OutR2, unsigned &InpR2) const {
  if (!BTP->has(OutR1) || !BTP->has(InpR1) || !BTP->has(OutR2))
    return false;
  const BitTracker::RegisterCell &OutC1 = BTP->lookup(OutR1);
  const BitTracker::RegisterCell &OutC2 = BTP->lookup(OutR2);
  unsigned W = OutC1.width();
  unsigned MatchR = 0;
  if (W != OutC2.width())
    return false;
  for (unsigned i = 0; i < W; ++i) {
    const BitTracker::BitValue &V1 = OutC1[i], &V2 = OutC2[i];
    if (V1.Type != V2.Type || V1.Type == BitTracker::BitValue::One)
      return false;
    if (V1.Type != BitTracker::BitValue::Ref)
      continue;
    if (V1.RefI.Pos != V2.RefI.Pos)
      return false;
    if (V1.RefI.Reg != InpR1)
      return false;
    if (V2.RefI.Reg == 0 || V2.RefI.Reg == OutR2)
      return false;
    if (!MatchR)
      MatchR = V2.RefI.Reg;
    else if (V2.RefI.Reg != MatchR)
      return false;
  }
  InpR2 = MatchR;
  return true;
}


void HexagonLoopRescheduling::moveGroup(InstrGroup &G, MachineBasicBlock &LB,
      MachineBasicBlock &PB, MachineBasicBlock::iterator At, unsigned OldPhiR,
      unsigned NewPredR) {
  DenseMap<unsigned,unsigned> RegMap;

  const TargetRegisterClass *PhiRC = MRI->getRegClass(NewPredR);
  unsigned PhiR = MRI->createVirtualRegister(PhiRC);
  BuildMI(LB, At, At->getDebugLoc(), HII->get(TargetOpcode::PHI), PhiR)
    .addReg(NewPredR)
    .addMBB(&PB)
    .addReg(G.Inp.Reg)
    .addMBB(&LB);
  RegMap.insert(std::make_pair(G.Inp.Reg, PhiR));

  for (unsigned i = G.Ins.size(); i > 0; --i) {
    const MachineInstr *SI = G.Ins[i-1];
    unsigned DR = getDefReg(SI);
    const TargetRegisterClass *RC = MRI->getRegClass(DR);
    unsigned NewDR = MRI->createVirtualRegister(RC);
    DebugLoc DL = SI->getDebugLoc();

    auto MIB = BuildMI(LB, At, DL, HII->get(SI->getOpcode()), NewDR);
    for (unsigned j = 0, m = SI->getNumOperands(); j < m; ++j) {
      const MachineOperand &Op = SI->getOperand(j);
      if (!Op.isReg()) {
        MIB.addOperand(Op);
        continue;
      }
      if (!Op.isUse())
        continue;
      unsigned UseR = RegMap[Op.getReg()];
      MIB.addReg(UseR, 0, Op.getSubReg());
    }
    RegMap.insert(std::make_pair(DR, NewDR));
  }

  HBS::replaceReg(OldPhiR, RegMap[G.Out.Reg], *MRI);
}


bool HexagonLoopRescheduling::processLoop(LoopCand &C) {
  DEBUG(dbgs() << "Processing loop in BB#" << C.LB->getNumber() << "\n");
  std::vector<PhiInfo> Phis;
  for (auto &I : *C.LB) {
    if (!I.isPHI())
      break;
    unsigned PR = getDefReg(&I);
    if (isConst(PR))
      continue;
    bool BadUse = false, GoodUse = false;
    for (auto UI = MRI->use_begin(PR), UE = MRI->use_end(); UI != UE; ++UI) {
      MachineInstr *UseI = UI->getParent();
      if (UseI->getParent() != C.LB) {
        BadUse = true;
        break;
      }
      if (isBitShuffle(UseI, PR) || isStoreInput(UseI, PR))
        GoodUse = true;
    }
    if (BadUse || !GoodUse)
      continue;

    Phis.push_back(PhiInfo(I, *C.LB));
  }

  DEBUG({
    dbgs() << "Phis: {";
    for (auto &I : Phis) {
      dbgs() << ' ' << PrintReg(I.DefR, HRI) << "=phi("
             << PrintReg(I.PR.Reg, HRI, I.PR.Sub) << ":b" << I.PB->getNumber()
             << ',' << PrintReg(I.LR.Reg, HRI, I.LR.Sub) << ":b"
             << I.LB->getNumber() << ')';
    }
    dbgs() << " }\n";
  });

  if (Phis.empty())
    return false;

  bool Changed = false;
  InstrList ShufIns;

  // Go backwards in the block: for each bit shuffling instruction, check
  // if that instruction could potentially be moved to the front of the loop:
  // the output of the loop cannot be used in a non-shuffling instruction
  // in this loop.
  for (auto I = C.LB->rbegin(), E = C.LB->rend(); I != E; ++I) {
    if (I->isTerminator())
      continue;
    if (I->isPHI())
      break;

    RegisterSet Defs;
    HBS::getInstrDefs(*I, Defs);
    if (Defs.count() != 1)
      continue;
    unsigned DefR = Defs.find_first();
    if (!TargetRegisterInfo::isVirtualRegister(DefR))
      continue;
    if (!isBitShuffle(&*I, DefR))
      continue;

    bool BadUse = false;
    for (auto UI = MRI->use_begin(DefR), UE = MRI->use_end(); UI != UE; ++UI) {
      MachineInstr *UseI = UI->getParent();
      if (UseI->getParent() == C.LB) {
        if (UseI->isPHI()) {
          // If the use is in a phi node in this loop, then it should be
          // the value corresponding to the back edge.
          unsigned Idx = UI.getOperandNo();
          if (UseI->getOperand(Idx+1).getMBB() != C.LB)
            BadUse = true;
        } else {
          auto F = std::find(ShufIns.begin(), ShufIns.end(), UseI);
          if (F == ShufIns.end())
            BadUse = true;
        }
      } else {
        // There is a use outside of the loop, but there is no epilog block
        // suitable for a copy-out.
        if (C.EB == nullptr)
          BadUse = true;
      }
      if (BadUse)
        break;
    }

    if (BadUse)
      continue;
    ShufIns.push_back(&*I);
  }

  // Partition the list of shuffling instructions into instruction groups,
  // where each group has to be moved as a whole (i.e. a group is a chain of
  // dependent instructions). A group produces a single live output register,
  // which is meant to be the input of the loop phi node (although this is
  // not checked here yet). It also uses a single register as its input,
  // which is some value produced in the loop body. After moving the group
  // to the beginning of the loop, that input register would need to be
  // the loop-carried register (through a phi node) instead of the (currently
  // loop-carried) output register.
  typedef std::vector<InstrGroup> InstrGroupList;
  InstrGroupList Groups;

  for (unsigned i = 0, n = ShufIns.size(); i < n; ++i) {
    MachineInstr *SI = ShufIns[i];
    if (SI == nullptr)
      continue;

    InstrGroup G;
    G.Ins.push_back(SI);
    G.Out.Reg = getDefReg(SI);
    RegisterSet Inputs;
    HBS::getInstrUses(*SI, Inputs);

    for (unsigned j = i+1; j < n; ++j) {
      MachineInstr *MI = ShufIns[j];
      if (MI == nullptr)
        continue;
      RegisterSet Defs;
      HBS::getInstrDefs(*MI, Defs);
      // If this instruction does not define any pending inputs, skip it.
      if (!Defs.intersects(Inputs))
        continue;
      // Otherwise, add it to the current group and remove the inputs that
      // are defined by MI.
      G.Ins.push_back(MI);
      Inputs.remove(Defs);
      // Then add all registers used by MI.
      HBS::getInstrUses(*MI, Inputs);
      ShufIns[j] = nullptr;
    }

    // Only add a group if it requires at most one register.
    if (Inputs.count() > 1)
      continue;
    auto LoopInpEq = [G] (const PhiInfo &P) -> bool {
      return G.Out.Reg == P.LR.Reg;
    };
    if (std::find_if(Phis.begin(), Phis.end(), LoopInpEq) == Phis.end())
      continue;

    G.Inp.Reg = Inputs.find_first();
    Groups.push_back(G);
  }

  DEBUG({
    for (unsigned i = 0, n = Groups.size(); i < n; ++i) {
      InstrGroup &G = Groups[i];
      dbgs() << "Group[" << i << "] inp: "
             << PrintReg(G.Inp.Reg, HRI, G.Inp.Sub)
             << "  out: " << PrintReg(G.Out.Reg, HRI, G.Out.Sub) << "\n";
      for (unsigned j = 0, m = G.Ins.size(); j < m; ++j)
        dbgs() << "  " << *G.Ins[j];
    }
  });

  for (unsigned i = 0, n = Groups.size(); i < n; ++i) {
    InstrGroup &G = Groups[i];
    if (!isShuffleOf(G.Out.Reg, G.Inp.Reg))
      continue;
    auto LoopInpEq = [G] (const PhiInfo &P) -> bool {
      return G.Out.Reg == P.LR.Reg;
    };
    auto F = std::find_if(Phis.begin(), Phis.end(), LoopInpEq);
    if (F == Phis.end())
      continue;
    unsigned PredR = 0;
    if (!isSameShuffle(G.Out.Reg, G.Inp.Reg, F->PR.Reg, PredR)) {
      const MachineInstr *DefPredR = MRI->getVRegDef(F->PR.Reg);
      unsigned Opc = DefPredR->getOpcode();
      if (Opc != Hexagon::A2_tfrsi && Opc != Hexagon::A2_tfrpi)
        continue;
      if (!DefPredR->getOperand(1).isImm())
        continue;
      if (DefPredR->getOperand(1).getImm() != 0)
        continue;
      const TargetRegisterClass *RC = MRI->getRegClass(G.Inp.Reg);
      if (RC != MRI->getRegClass(F->PR.Reg)) {
        PredR = MRI->createVirtualRegister(RC);
        unsigned TfrI = (RC == &Hexagon::IntRegsRegClass) ? Hexagon::A2_tfrsi
                                                          : Hexagon::A2_tfrpi;
        auto T = C.PB->getFirstTerminator();
        DebugLoc DL = (T != C.PB->end()) ? T->getDebugLoc() : DebugLoc();
        BuildMI(*C.PB, T, DL, HII->get(TfrI), PredR)
          .addImm(0);
      } else {
        PredR = F->PR.Reg;
      }
    }
    assert(MRI->getRegClass(PredR) == MRI->getRegClass(G.Inp.Reg));
    moveGroup(G, *F->LB, *F->PB, F->LB->getFirstNonPHI(), F->DefR, PredR);
    Changed = true;
  }

  return Changed;
}


bool HexagonLoopRescheduling::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  auto &HST = MF.getSubtarget<HexagonSubtarget>();
  HII = HST.getInstrInfo();
  HRI = HST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  const HexagonEvaluator HE(*HRI, *MRI, *HII, MF);
  BitTracker BT(HE, MF);
  DEBUG(BT.trace(true));
  BT.run();
  BTP = &BT;

  std::vector<LoopCand> Cand;

  for (auto &B : MF) {
    if (B.pred_size() != 2 || B.succ_size() != 2)
      continue;
    MachineBasicBlock *PB = nullptr;
    bool IsLoop = false;
    for (auto PI = B.pred_begin(), PE = B.pred_end(); PI != PE; ++PI) {
      if (*PI != &B)
        PB = *PI;
      else
        IsLoop = true;
    }
    if (!IsLoop)
      continue;

    MachineBasicBlock *EB = nullptr;
    for (auto SI = B.succ_begin(), SE = B.succ_end(); SI != SE; ++SI) {
      if (*SI == &B)
        continue;
      // Set EP to the epilog block, if it has only 1 predecessor (i.e. the
      // edge from B to EP is non-critical.
      if ((*SI)->pred_size() == 1)
        EB = *SI;
      break;
    }

    Cand.push_back(LoopCand(&B, PB, EB));
  }

  bool Changed = false;
  for (auto &C : Cand)
    Changed |= processLoop(C);

  return Changed;
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createHexagonLoopRescheduling() {
  return new HexagonLoopRescheduling();
}

FunctionPass *llvm::createHexagonBitSimplify() {
  return new HexagonBitSimplify();
}


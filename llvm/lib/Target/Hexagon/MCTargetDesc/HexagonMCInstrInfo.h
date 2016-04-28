//===- HexagonMCInstrInfo.cpp - Utility functions on Hexagon MCInsts ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utility functions for Hexagon specific MCInst queries
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCINSTRINFO_H
#define LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCINSTRINFO_H

#include "HexagonMCExpr.h"
#include "llvm/MC/MCInst.h"

namespace llvm {
class HexagonMCChecker;
class MCContext;
class MCInstrDesc;
class MCInstrInfo;
class MCInst;
class MCOperand;
class MCSubtargetInfo;
namespace HexagonII {
enum class MemAccessSize;
}
class DuplexCandidate {
public:
  unsigned packetIndexI, packetIndexJ, iClass;
  DuplexCandidate(unsigned i, unsigned j, unsigned iClass)
      : packetIndexI(i), packetIndexJ(j), iClass(iClass) {}
};
namespace HexagonMCInstrInfo {
size_t const innerLoopOffset = 0;
int64_t const innerLoopMask = 1 << innerLoopOffset;

size_t const outerLoopOffset = 1;
int64_t const outerLoopMask = 1 << outerLoopOffset;

// do not reorder memory load/stores by default load/stores are re-ordered
// and by default loads can be re-ordered
size_t const memReorderDisabledOffset = 2;
int64_t const memReorderDisabledMask = 1 << memReorderDisabledOffset;

// allow re-ordering of memory stores by default stores cannot be re-ordered
size_t const memStoreReorderEnabledOffset = 3;
int64_t const memStoreReorderEnabledMask = 1 << memStoreReorderEnabledOffset;

size_t const bundleInstructionsOffset = 1;

void addConstant(MCInst &MI, uint64_t Value, MCContext &Context);
void addConstExtender(MCContext &Context, MCInstrInfo const &MCII, MCInst &MCB,
                      MCInst const &MCI);

// Returns a iterator range of instructions in this bundle
iterator_range<MCInst::const_iterator> bundleInstructions(MCInst const &MCI);

// Returns the number of instructions in the bundle
size_t bundleSize(MCInst const &MCI);

// Put the packet in to canonical form, compound, duplex, pad, and shuffle
bool canonicalizePacket(MCInstrInfo const &MCII, MCSubtargetInfo const &STI,
                        MCContext &Context, MCInst &MCB,
                        HexagonMCChecker *Checker);

// Clamp off upper 26 bits of extendable operand for emission
void clampExtended(MCInstrInfo const &MCII, MCContext &Context, MCInst &MCI);

MCInst createBundle();

// Return the extender for instruction at Index or nullptr if none
MCInst const *extenderForIndex(MCInst const &MCB, size_t Index);
void extendIfNeeded(MCContext &Context, MCInstrInfo const &MCII, MCInst &MCB,
                    MCInst const &MCI);

// Create a duplex instruction given the two subinsts
MCInst *deriveDuplex(MCContext &Context, unsigned iClass, MCInst const &inst0,
                     MCInst const &inst1);
MCInst deriveExtender(MCInstrInfo const &MCII, MCInst const &Inst,
                      MCOperand const &MO);

// Convert this instruction in to a duplex subinst
MCInst deriveSubInst(MCInst const &Inst);

// Return the extender for instruction at Index or nullptr if none
MCInst const *extenderForIndex(MCInst const &MCB, size_t Index);

// Return memory access size
HexagonII::MemAccessSize getAccessSize(MCInstrInfo const &MCII,
                                       MCInst const &MCI);

// Return number of bits in the constant extended operand.
unsigned getBitCount(MCInstrInfo const &MCII, MCInst const &MCI);

// Return constant extended operand number.
unsigned short getCExtOpNum(MCInstrInfo const &MCII, MCInst const &MCI);

MCInstrDesc const &getDesc(MCInstrInfo const &MCII, MCInst const &MCI);

// Return which duplex group this instruction belongs to
unsigned getDuplexCandidateGroup(MCInst const &MI);

// Return a list of all possible instruction duplex combinations
SmallVector<DuplexCandidate, 8> getDuplexPossibilties(MCInstrInfo const &MCII,
                                                      MCInst const &MCB);
unsigned getDuplexRegisterNumbering(unsigned Reg);

MCExpr const &getExpr(MCExpr const &Expr);

// Return the index of the extendable operand
unsigned short getExtendableOp(MCInstrInfo const &MCII, MCInst const &MCI);

// Return a reference to the extendable operand
MCOperand const &getExtendableOperand(MCInstrInfo const &MCII,
                                      MCInst const &MCI);

// Return the implicit alignment of the extendable operand
unsigned getExtentAlignment(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the number of logical bits of the extendable operand
unsigned getExtentBits(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the max value that a constant extendable operand can have
// without being extended.
int getMaxValue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the min value that a constant extendable operand can have
// without being extended.
int getMinValue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return instruction name
char const *getName(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the operand index for the new value.
unsigned short getNewValueOp(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the operand that consumes or produces a new value.
MCOperand const &getNewValueOperand(MCInstrInfo const &MCII, MCInst const &MCI);
unsigned short getNewValueOp2(MCInstrInfo const &MCII, MCInst const &MCI);
MCOperand const &getNewValueOperand2(MCInstrInfo const &MCII,
                                     MCInst const &MCI);

int getSubTarget(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the Hexagon ISA class for the insn.
unsigned getType(MCInstrInfo const &MCII, MCInst const &MCI);

/// Return the slots used by the insn.
unsigned getUnits(MCInstrInfo const &MCII, MCSubtargetInfo const &STI,
                  MCInst const &MCI);

// Does the packet have an extender for the instruction at Index
bool hasExtenderForIndex(MCInst const &MCB, size_t Index);

bool hasImmExt(MCInst const &MCI);

// Return whether the instruction is a legal new-value producer.
bool hasNewValue(MCInstrInfo const &MCII, MCInst const &MCI);
bool hasNewValue2(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the instruction at Index
MCInst const &instruction(MCInst const &MCB, size_t Index);

// Returns whether this MCInst is a wellformed bundle
bool isBundle(MCInst const &MCI);

// Return whether the insn is an actual insn.
bool isCanon(MCInstrInfo const &MCII, MCInst const &MCI);
bool isCompound(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the duplex iclass given the two duplex classes
unsigned iClassOfDuplexPair(unsigned Ga, unsigned Gb);

int64_t minConstant(MCInst const &MCI, size_t Index);
template <unsigned N, unsigned S>
bool inRange(MCInst const &MCI, size_t Index) {
  return isShiftedUInt<N, S>(minConstant(MCI, Index));
}
template <unsigned N, unsigned S>
bool inSRange(MCInst const &MCI, size_t Index) {
  return isShiftedInt<N, S>(minConstant(MCI, Index));
}
template <unsigned N> bool inRange(MCInst const &MCI, size_t Index) {
  return isUInt<N>(minConstant(MCI, Index));
}

// Return whether the instruction needs to be constant extended.
bool isConstExtended(MCInstrInfo const &MCII, MCInst const &MCI);

// Is this double register suitable for use in a duplex subinst
bool isDblRegForSubInst(unsigned Reg);

// Is this a duplex instruction
bool isDuplex(MCInstrInfo const &MCII, MCInst const &MCI);

// Can these instructions be duplexed
bool isDuplexPair(MCInst const &MIa, MCInst const &MIb);

// Can these duplex classes be combine in to a duplex instruction
bool isDuplexPairMatch(unsigned Ga, unsigned Gb);

// Return true if the insn may be extended based on the operand value.
bool isExtendable(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the instruction must be always extended.
bool isExtended(MCInstrInfo const &MCII, MCInst const &MCI);

/// Return whether it is a floating-point insn.
bool isFloat(MCInstrInfo const &MCII, MCInst const &MCI);

// Returns whether this instruction is an immediate extender
bool isImmext(MCInst const &MCI);

// Returns whether this bundle is an endloop0
bool isInnerLoop(MCInst const &MCI);

// Is this an integer register
bool isIntReg(unsigned Reg);

// Is this register suitable for use in a duplex subinst
bool isIntRegForSubInst(unsigned Reg);
bool isMemReorderDisabled(MCInst const &MCI);
bool isMemStoreReorderEnabled(MCInst const &MCI);

// Return whether the insn is a new-value consumer.
bool isNewValue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return true if the operand can be constant extended.
bool isOperandExtended(MCInstrInfo const &MCII, MCInst const &MCI,
                       unsigned short OperandNum);

// Can these two instructions be duplexed
bool isOrderedDuplexPair(MCInstrInfo const &MCII, MCInst const &MIa,
                         bool ExtendedA, MCInst const &MIb, bool ExtendedB,
                         bool bisReversable);

// Returns whether this bundle is an endloop1
bool isOuterLoop(MCInst const &MCI);

// Return whether this instruction is predicated
bool isPredicated(MCInstrInfo const &MCII, MCInst const &MCI);
bool isPredicateLate(MCInstrInfo const &MCII, MCInst const &MCI);
bool isPredicatedNew(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the predicate sense is true
bool isPredicatedTrue(MCInstrInfo const &MCII, MCInst const &MCI);

// Is this a predicate register
bool isPredReg(unsigned Reg);

// Return whether the insn is a prefix.
bool isPrefix(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the insn is solo, i.e., cannot be in a packet.
bool isSolo(MCInstrInfo const &MCII, MCInst const &MCI);

/// Return whether the insn can be packaged only with A and X-type insns.
bool isSoloAX(MCInstrInfo const &MCII, MCInst const &MCI);

/// Return whether the insn can be packaged only with an A-type insn in slot #1.
bool isSoloAin1(MCInstrInfo const &MCII, MCInst const &MCI);
bool isSubInstruction(MCInst const &MCI);
bool isVector(MCInstrInfo const &MCII, MCInst const &MCI);
bool mustExtend(MCExpr const &Expr);
bool mustNotExtend(MCExpr const &Expr);

// Pad the bundle with nops to satisfy endloop requirements
void padEndloop(MCContext &Context, MCInst &MCI);

bool prefersSlot3(MCInstrInfo const &MCII, MCInst const &MCI);

// Replace the instructions inside MCB, represented by Candidate
void replaceDuplex(MCContext &Context, MCInst &MCB, DuplexCandidate Candidate);

bool s23_2_reloc(MCExpr const &Expr);
// Marks a bundle as endloop0
void setInnerLoop(MCInst &MCI);
void setMemReorderDisabled(MCInst &MCI);
void setMemStoreReorderEnabled(MCInst &MCI);
void setMustExtend(MCExpr const &Expr, bool Val = true);
void setMustNotExtend(MCExpr const &Expr, bool Val = true);
void setS23_2_reloc(MCExpr const &Expr, bool Val = true);

// Marks a bundle as endloop1
void setOuterLoop(MCInst &MCI);

// Would duplexing this instruction create a requirement to extend
bool subInstWouldBeExtended(MCInst const &potentialDuplex);
unsigned SubregisterBit(unsigned Consumer, unsigned Producer,
                        unsigned Producer2);

// Attempt to find and replace compound pairs
void tryCompound(MCInstrInfo const &MCII, MCContext &Context, MCInst &MCI);
}
}

#endif // LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCINSTRINFO_H

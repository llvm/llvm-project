//===-- MipsTargetStreamer.h - Mips Target Streamer ------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSTARGETSTREAMER_H
#define MIPSTARGETSTREAMER_H

#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCStreamer.h"
#include "MCTargetDesc/MipsABIFlagsSection.h"

namespace llvm {

struct MipsABIFlagsSection;

class MipsTargetStreamer : public MCTargetStreamer {
public:
  MipsTargetStreamer(MCStreamer &S);
  virtual void emitDirectiveSetMicroMips();
  virtual void emitDirectiveSetNoMicroMips();
  virtual void emitDirectiveSetMips16();
  virtual void emitDirectiveSetNoMips16();

  virtual void emitDirectiveSetReorder();
  virtual void emitDirectiveSetNoReorder();
  virtual void emitDirectiveSetMacro();
  virtual void emitDirectiveSetNoMacro();
  virtual void emitDirectiveSetAt();
  virtual void emitDirectiveSetNoAt();
  virtual void emitDirectiveEnd(StringRef Name);

  virtual void emitDirectiveEnt(const MCSymbol &Symbol);
  virtual void emitDirectiveAbiCalls();
  virtual void emitDirectiveNaN2008();
  virtual void emitDirectiveNaNLegacy();
  virtual void emitDirectiveOptionPic0();
  virtual void emitDirectiveOptionPic2();
  virtual void emitFrame(unsigned StackReg, unsigned StackSize,
                         unsigned ReturnReg);
  virtual void emitMask(unsigned CPUBitmask, int CPUTopSavedRegOff);
  virtual void emitFMask(unsigned FPUBitmask, int FPUTopSavedRegOff);

  virtual void emitDirectiveSetMips32R2();
  virtual void emitDirectiveSetMips64();
  virtual void emitDirectiveSetMips64R2();
  virtual void emitDirectiveSetDsp();
  virtual void emitDirectiveSetPop();
  virtual void emitDirectiveSetPush();

  // PIC support
  virtual void emitDirectiveCpload(unsigned RegNo);
  virtual void emitDirectiveCpsetup(unsigned RegNo, int RegOrOffset,
                                    const MCSymbol &Sym, bool IsReg);

  /// Emit a '.module fp=value' directive using the given values.
  /// Updates the .MIPS.abiflags section
  virtual void emitDirectiveModuleFP(MipsABIFlagsSection::FpABIKind Value,
                                     bool Is32BitABI) {
    ABIFlagsSection.setFpABI(Value, Is32BitABI);
  }

  /// Emit a '.module fp=value' directive using the current values of the
  /// .MIPS.abiflags section.
  void emitDirectiveModuleFP() {
    emitDirectiveModuleFP(ABIFlagsSection.getFpABI(),
                          ABIFlagsSection.Is32BitABI);
  }

  virtual void emitDirectiveModuleOddSPReg(bool Enabled, bool IsO32ABI);
  virtual void emitDirectiveSetFp(MipsABIFlagsSection::FpABIKind Value){};
  virtual void emitMipsAbiFlags(){};
  void setCanHaveModuleDir(bool Can) { canHaveModuleDirective = Can; }
  bool getCanHaveModuleDir() { return canHaveModuleDirective; }

  // This method enables template classes to set internal abi flags
  // structure values.
  template <class PredicateLibrary>
  void updateABIInfo(const PredicateLibrary &P) {
    ABIFlagsSection.setAllFromPredicates(P);
  }

  MipsABIFlagsSection &getABIFlagsSection() { return ABIFlagsSection; }

protected:
  MipsABIFlagsSection ABIFlagsSection;

private:
  bool canHaveModuleDirective;
};

// This part is for ascii assembly output
class MipsTargetAsmStreamer : public MipsTargetStreamer {
  formatted_raw_ostream &OS;

public:
  MipsTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);
  void emitDirectiveSetMicroMips() override;
  void emitDirectiveSetNoMicroMips() override;
  void emitDirectiveSetMips16() override;
  void emitDirectiveSetNoMips16() override;

  void emitDirectiveSetReorder() override;
  void emitDirectiveSetNoReorder() override;
  void emitDirectiveSetMacro() override;
  void emitDirectiveSetNoMacro() override;
  void emitDirectiveSetAt() override;
  void emitDirectiveSetNoAt() override;
  void emitDirectiveEnd(StringRef Name) override;

  void emitDirectiveEnt(const MCSymbol &Symbol) override;
  void emitDirectiveAbiCalls() override;
  void emitDirectiveNaN2008() override;
  void emitDirectiveNaNLegacy() override;
  void emitDirectiveOptionPic0() override;
  void emitDirectiveOptionPic2() override;
  void emitFrame(unsigned StackReg, unsigned StackSize,
                 unsigned ReturnReg) override;
  void emitMask(unsigned CPUBitmask, int CPUTopSavedRegOff) override;
  void emitFMask(unsigned FPUBitmask, int FPUTopSavedRegOff) override;

  void emitDirectiveSetMips32R2() override;
  void emitDirectiveSetMips64() override;
  void emitDirectiveSetMips64R2() override;
  void emitDirectiveSetDsp() override;
  void emitDirectiveSetPop() override;
  void emitDirectiveSetPush() override;

  // PIC support
  virtual void emitDirectiveCpload(unsigned RegNo);
  void emitDirectiveCpsetup(unsigned RegNo, int RegOrOffset,
                            const MCSymbol &Sym, bool IsReg) override;

  // ABI Flags
  void emitDirectiveModuleFP(MipsABIFlagsSection::FpABIKind Value,
                             bool Is32BitABI) override;
  void emitDirectiveModuleOddSPReg(bool Enabled, bool IsO32ABI) override;
  void emitDirectiveSetFp(MipsABIFlagsSection::FpABIKind Value) override;
  void emitMipsAbiFlags() override;
};

// This part is for ELF object output
class MipsTargetELFStreamer : public MipsTargetStreamer {
  bool MicroMipsEnabled;
  const MCSubtargetInfo &STI;
  bool Pic;

public:
  bool isMicroMipsEnabled() const { return MicroMipsEnabled; }
  MCELFStreamer &getStreamer();
  MipsTargetELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);

  void emitLabel(MCSymbol *Symbol) override;
  void emitAssignment(MCSymbol *Symbol, const MCExpr *Value) override;
  void finish() override;

  void emitDirectiveSetMicroMips() override;
  void emitDirectiveSetNoMicroMips() override;
  void emitDirectiveSetMips16() override;
  void emitDirectiveSetNoMips16() override;

  void emitDirectiveSetReorder() override;
  void emitDirectiveSetNoReorder() override;
  void emitDirectiveSetMacro() override;
  void emitDirectiveSetNoMacro() override;
  void emitDirectiveSetAt() override;
  void emitDirectiveSetNoAt() override;
  void emitDirectiveEnd(StringRef Name) override;

  void emitDirectiveEnt(const MCSymbol &Symbol) override;
  void emitDirectiveAbiCalls() override;
  void emitDirectiveNaN2008() override;
  void emitDirectiveNaNLegacy() override;
  void emitDirectiveOptionPic0() override;
  void emitDirectiveOptionPic2() override;
  void emitFrame(unsigned StackReg, unsigned StackSize,
                 unsigned ReturnReg) override;
  void emitMask(unsigned CPUBitmask, int CPUTopSavedRegOff) override;
  void emitFMask(unsigned FPUBitmask, int FPUTopSavedRegOff) override;

  void emitDirectiveSetMips32R2() override;
  void emitDirectiveSetMips64() override;
  void emitDirectiveSetMips64R2() override;
  void emitDirectiveSetDsp() override;

  // PIC support
  virtual void emitDirectiveCpload(unsigned RegNo);
  void emitDirectiveCpsetup(unsigned RegNo, int RegOrOffset,
                            const MCSymbol &Sym, bool IsReg) override;

  // ABI Flags
  void emitDirectiveModuleOddSPReg(bool Enabled, bool IsO32ABI) override;
  void emitMipsAbiFlags() override;

protected:
  bool isO32() const { return STI.getFeatureBits() & Mips::FeatureO32; }
  bool isN32() const { return STI.getFeatureBits() & Mips::FeatureN32; }
  bool isN64() const { return STI.getFeatureBits() & Mips::FeatureN64; }
};
}
#endif

//===-- llvm/Target/TargetLoweringObjectFile.h - Object Info ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements classes used to handle lowerings specific to common
// object file formats.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETLOWERINGOBJECTFILE_H
#define LLVM_TARGET_TARGETLOWERINGOBJECTFILE_H

#include "llvm/Module.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/ADT/ArrayRef.h"

namespace llvm {
  class MachineModuleInfo;
  class Mangler;
  class MCContext;
  class MCExpr;
  class MCSection;
  class MCSymbol;
  class MCStreamer;
  class GlobalValue;
  class TargetMachine;
  
class TargetLoweringObjectFile : public MCObjectFileInfo {
  MCContext *Ctx;

  TargetLoweringObjectFile(
    const TargetLoweringObjectFile&) LLVM_DELETED_FUNCTION;
  void operator=(const TargetLoweringObjectFile&) LLVM_DELETED_FUNCTION;

public:
  MCContext &getContext() const { return *Ctx; }

  TargetLoweringObjectFile() : MCObjectFileInfo(), Ctx(0) {}
  
  virtual ~TargetLoweringObjectFile();
  
  /// Initialize - this method must be called before any actual lowering is
  /// done.  This specifies the current context for codegen, and gives the
  /// lowering implementations a chance to set up their default sections.
  virtual void Initialize(MCContext &ctx, const TargetMachine &TM);
  
  virtual void emitPersonalityValue(MCStreamer &Streamer,
                                    const TargetMachine &TM,
                                    const MCSymbol *Sym) const;

  /// emitModuleFlags - Emit the module flags that the platform cares about.
  virtual void emitModuleFlags(MCStreamer &,
                               ArrayRef<Module::ModuleFlagEntry>,
                               Mangler *, const TargetMachine &) const {
  }

  /// shouldEmitUsedDirectiveFor - This hook allows targets to selectively
  /// decide not to emit the UsedDirective for some symbols in llvm.used.
  /// FIXME: REMOVE this (rdar://7071300)
  virtual bool shouldEmitUsedDirectiveFor(const GlobalValue *GV,
                                          Mangler *) const {
    return GV != 0;
  }
  
  /// getSectionForConstant - Given a constant with the SectionKind, return a
  /// section that it should be placed in.
  virtual const MCSection *getSectionForConstant(SectionKind Kind) const;
  
  /// getKindForGlobal - Classify the specified global variable into a set of
  /// target independent categories embodied in SectionKind.
  static SectionKind getKindForGlobal(const GlobalValue *GV,
                                      const TargetMachine &TM);
  
  /// SectionForGlobal - This method computes the appropriate section to emit
  /// the specified global variable or function definition.  This should not
  /// be passed external (or available externally) globals.
  const MCSection *SectionForGlobal(const GlobalValue *GV,
                                    SectionKind Kind, Mangler *Mang,
                                    const TargetMachine &TM) const;
  
  /// SectionForGlobal - This method computes the appropriate section to emit
  /// the specified global variable or function definition.  This should not
  /// be passed external (or available externally) globals.
  const MCSection *SectionForGlobal(const GlobalValue *GV,
                                    Mangler *Mang,
                                    const TargetMachine &TM) const {
    return SectionForGlobal(GV, getKindForGlobal(GV, TM), Mang, TM);
  }

  /// getExplicitSectionGlobal - Targets should implement this method to assign
  /// a section to globals with an explicit section specfied.  The
  /// implementation of this method can assume that GV->hasSection() is true.
  virtual const MCSection *
  getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind, 
                           Mangler *Mang, const TargetMachine &TM) const = 0;
  
  /// getSpecialCasedSectionGlobals - Allow the target to completely override
  /// section assignment of a global.
  virtual const MCSection *
  getSpecialCasedSectionGlobals(const GlobalValue *GV, Mangler *Mang,
                                SectionKind Kind) const {
    return 0;
  }
  
  /// getExprForDwarfGlobalReference - Return an MCExpr to use for a reference
  /// to the specified global variable from exception handling information.
  ///
  virtual const MCExpr *
  getExprForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                                 MachineModuleInfo *MMI, unsigned Encoding,
                                 MCStreamer &Streamer) const;

  // getCFIPersonalitySymbol - The symbol that gets passed to .cfi_personality.
  virtual MCSymbol *
  getCFIPersonalitySymbol(const GlobalValue *GV, Mangler *Mang,
                          MachineModuleInfo *MMI) const;

  /// 
  const MCExpr *
  getExprForDwarfReference(const MCSymbol *Sym, unsigned Encoding,
                           MCStreamer &Streamer) const;

  virtual const MCSection *
  getStaticCtorSection(unsigned Priority = 65535) const {
    (void)Priority;
    return StaticCtorSection;
  }
  virtual const MCSection *
  getStaticDtorSection(unsigned Priority = 65535) const {
    (void)Priority;
    return StaticDtorSection;
  }

protected:
  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
};

} // end namespace llvm

#endif

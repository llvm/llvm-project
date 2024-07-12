//===- MCAssembler.h - Object File Generation -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASSEMBLER_H
#define LLVM_MC_MCASSEMBLER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCLinkerOptimizationHint.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/VersionTuple.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace llvm {

class MCBoundaryAlignFragment;
class MCCVDefRangeFragment;
class MCCVInlineLineTableFragment;
class MCDwarfCallFrameFragment;
class MCDwarfLineAddrFragment;
class MCEncodedFragment;
class MCFixup;
class MCLEBFragment;
class MCPseudoProbeAddrFragment;
class MCRelaxableFragment;
class MCSymbolRefExpr;
class raw_ostream;
class MCAsmBackend;
class MCContext;
class MCCodeEmitter;
class MCFragment;
class MCObjectWriter;
class MCSection;
class MCValue;

class MCAssembler {
public:
  using SectionListType = SmallVector<MCSection *, 0>;
  using const_iterator = pointee_iterator<SectionListType::const_iterator>;

  /// MachO specific deployment target version info.
  // A Major version of 0 indicates that no version information was supplied
  // and so the corresponding load command should not be emitted.
  using VersionInfoType = struct {
    bool EmitBuildVersion;
    union {
      MCVersionMinType Type;          ///< Used when EmitBuildVersion==false.
      MachO::PlatformType Platform;   ///< Used when EmitBuildVersion==true.
    } TypeOrPlatform;
    unsigned Major;
    unsigned Minor;
    unsigned Update;
    /// An optional version of the SDK that was used to build the source.
    VersionTuple SDKVersion;
  };

private:
  MCContext &Context;

  std::unique_ptr<MCAsmBackend> Backend;
  std::unique_ptr<MCCodeEmitter> Emitter;
  std::unique_ptr<MCObjectWriter> Writer;

  bool HasLayout = false;
  bool RelaxAll = false;
  bool SubsectionsViaSymbols = false;
  bool IncrementalLinkerCompatible = false;

  SectionListType Sections;

  SmallVector<const MCSymbol *, 0> Symbols;

  /// The list of linker options to propagate into the object file.
  std::vector<std::vector<std::string>> LinkerOptions;

  /// List of declared file names
  std::vector<std::pair<std::string, size_t>> FileNames;
  // Optional compiler version.
  std::string CompilerVersion;

  MCDwarfLineTableParams LTParams;

  /// The set of function symbols for which a .thumb_func directive has
  /// been seen.
  //
  // FIXME: We really would like this in target specific code rather than
  // here. Maybe when the relocation stuff moves to target specific,
  // this can go with it? The streamer would need some target specific
  // refactoring too.
  mutable SmallPtrSet<const MCSymbol *, 32> ThumbFuncs;

  /// The bundle alignment size currently set in the assembler.
  ///
  /// By default it's 0, which means bundling is disabled.
  unsigned BundleAlignSize = 0;

  /// ELF specific e_header flags
  // It would be good if there were an MCELFAssembler class to hold this.
  // ELF header flags are used both by the integrated and standalone assemblers.
  // Access to the flags is necessary in cases where assembler directives affect
  // which flags to be set.
  unsigned ELFHeaderEFlags = 0;

  /// Used to communicate Linker Optimization Hint information between
  /// the Streamer and the .o writer
  MCLOHContainer LOHContainer;

  VersionInfoType VersionInfo;
  VersionInfoType DarwinTargetVariantVersionInfo;

  /// Evaluate a fixup to a relocatable expression and the value which should be
  /// placed into the fixup.
  ///
  /// \param Fixup The fixup to evaluate.
  /// \param DF The fragment the fixup is inside.
  /// \param Target [out] On return, the relocatable expression the fixup
  /// evaluates to.
  /// \param Value [out] On return, the value of the fixup as currently laid
  /// out.
  /// \param WasForced [out] On return, the value in the fixup is set to the
  /// correct value if WasForced is true, even if evaluateFixup returns false.
  /// \return Whether the fixup value was fully resolved. This is true if the
  /// \p Value result is fixed, otherwise the value may change due to
  /// relocation.
  bool evaluateFixup(const MCFixup &Fixup, const MCFragment *DF,
                     MCValue &Target, const MCSubtargetInfo *STI,
                     uint64_t &Value, bool &WasForced) const;

  /// Check whether a fixup can be satisfied, or whether it needs to be relaxed
  /// (increased in size, in order to hold its value correctly).
  bool fixupNeedsRelaxation(const MCFixup &Fixup, const MCRelaxableFragment *DF) const;

  /// Check whether the given fragment needs relaxation.
  bool fragmentNeedsRelaxation(const MCRelaxableFragment *IF) const;

  /// Perform one layout iteration and return true if any offsets
  /// were adjusted.
  bool layoutOnce();

  /// Perform relaxation on a single fragment - returns true if the fragment
  /// changes as a result of relaxation.
  bool relaxFragment(MCFragment &F);
  bool relaxInstruction(MCRelaxableFragment &IF);
  bool relaxLEB(MCLEBFragment &IF);
  bool relaxBoundaryAlign(MCBoundaryAlignFragment &BF);
  bool relaxDwarfLineAddr(MCDwarfLineAddrFragment &DF);
  bool relaxDwarfCallFrameFragment(MCDwarfCallFrameFragment &DF);
  bool relaxCVInlineLineTable(MCCVInlineLineTableFragment &DF);
  bool relaxCVDefRange(MCCVDefRangeFragment &DF);
  bool relaxPseudoProbeAddr(MCPseudoProbeAddrFragment &DF);

  std::tuple<MCValue, uint64_t, bool>
  handleFixup(MCFragment &F, const MCFixup &Fixup, const MCSubtargetInfo *STI);

public:
  struct Symver {
    SMLoc Loc;
    const MCSymbol *Sym;
    StringRef Name;
    // True if .symver *, *@@@* or .symver *, *, remove.
    bool KeepOriginalSym;
  };
  std::vector<Symver> Symvers;

  /// Construct a new assembler instance.
  //
  // FIXME: How are we going to parameterize this? Two obvious options are stay
  // concrete and require clients to pass in a target like object. The other
  // option is to make this abstract, and have targets provide concrete
  // implementations as we do with AsmParser.
  MCAssembler(MCContext &Context, std::unique_ptr<MCAsmBackend> Backend,
              std::unique_ptr<MCCodeEmitter> Emitter,
              std::unique_ptr<MCObjectWriter> Writer);
  MCAssembler(const MCAssembler &) = delete;
  MCAssembler &operator=(const MCAssembler &) = delete;
  ~MCAssembler();

  /// Compute the effective fragment size.
  uint64_t computeFragmentSize(const MCFragment &F) const;

  void layoutBundle(MCFragment *Prev, MCFragment *F) const;
  void ensureValid(MCSection &Sec) const;

  // Get the offset of the given fragment inside its containing section.
  uint64_t getFragmentOffset(const MCFragment &F) const;

  uint64_t getSectionAddressSize(const MCSection &Sec) const;
  uint64_t getSectionFileSize(const MCSection &Sec) const;

  // Get the offset of the given symbol, as computed in the current
  // layout.
  // \return True on success.
  bool getSymbolOffset(const MCSymbol &S, uint64_t &Val) const;

  // Variant that reports a fatal error if the offset is not computable.
  uint64_t getSymbolOffset(const MCSymbol &S) const;

  // If this symbol is equivalent to A + Constant, return A.
  const MCSymbol *getBaseSymbol(const MCSymbol &Symbol) const;

  /// Emit the section contents to \p OS.
  void writeSectionData(raw_ostream &OS, const MCSection *Section) const;

  /// Check whether a given symbol has been flagged with .thumb_func.
  bool isThumbFunc(const MCSymbol *Func) const;

  /// Flag a function symbol as the target of a .thumb_func directive.
  void setIsThumbFunc(const MCSymbol *Func) { ThumbFuncs.insert(Func); }

  /// ELF e_header flags
  unsigned getELFHeaderEFlags() const { return ELFHeaderEFlags; }
  void setELFHeaderEFlags(unsigned Flags) { ELFHeaderEFlags = Flags; }

  /// MachO deployment target version information.
  const VersionInfoType &getVersionInfo() const { return VersionInfo; }
  void setVersionMin(MCVersionMinType Type, unsigned Major, unsigned Minor,
                     unsigned Update,
                     VersionTuple SDKVersion = VersionTuple()) {
    VersionInfo.EmitBuildVersion = false;
    VersionInfo.TypeOrPlatform.Type = Type;
    VersionInfo.Major = Major;
    VersionInfo.Minor = Minor;
    VersionInfo.Update = Update;
    VersionInfo.SDKVersion = SDKVersion;
  }
  void setBuildVersion(MachO::PlatformType Platform, unsigned Major,
                       unsigned Minor, unsigned Update,
                       VersionTuple SDKVersion = VersionTuple()) {
    VersionInfo.EmitBuildVersion = true;
    VersionInfo.TypeOrPlatform.Platform = Platform;
    VersionInfo.Major = Major;
    VersionInfo.Minor = Minor;
    VersionInfo.Update = Update;
    VersionInfo.SDKVersion = SDKVersion;
  }

  const VersionInfoType &getDarwinTargetVariantVersionInfo() const {
    return DarwinTargetVariantVersionInfo;
  }
  void setDarwinTargetVariantBuildVersion(MachO::PlatformType Platform,
                                          unsigned Major, unsigned Minor,
                                          unsigned Update,
                                          VersionTuple SDKVersion) {
    DarwinTargetVariantVersionInfo.EmitBuildVersion = true;
    DarwinTargetVariantVersionInfo.TypeOrPlatform.Platform = Platform;
    DarwinTargetVariantVersionInfo.Major = Major;
    DarwinTargetVariantVersionInfo.Minor = Minor;
    DarwinTargetVariantVersionInfo.Update = Update;
    DarwinTargetVariantVersionInfo.SDKVersion = SDKVersion;
  }

  /// Reuse an assembler instance
  ///
  void reset();

  MCContext &getContext() const { return Context; }

  MCAsmBackend *getBackendPtr() const { return Backend.get(); }

  MCCodeEmitter *getEmitterPtr() const { return Emitter.get(); }

  MCObjectWriter *getWriterPtr() const { return Writer.get(); }

  MCAsmBackend &getBackend() const { return *Backend; }

  MCCodeEmitter &getEmitter() const { return *Emitter; }

  MCObjectWriter &getWriter() const { return *Writer; }

  MCDwarfLineTableParams getDWARFLinetableParams() const { return LTParams; }
  void setDWARFLinetableParams(MCDwarfLineTableParams P) { LTParams = P; }

  /// Finish - Do final processing and write the object to the output stream.
  /// \p Writer is used for custom object writer (as the MCJIT does),
  /// if not specified it is automatically created from backend.
  void Finish();

  // Layout all section and prepare them for emission.
  void layout();

  // FIXME: This does not belong here.
  bool getSubsectionsViaSymbols() const { return SubsectionsViaSymbols; }
  void setSubsectionsViaSymbols(bool Value) { SubsectionsViaSymbols = Value; }

  bool isIncrementalLinkerCompatible() const {
    return IncrementalLinkerCompatible;
  }
  void setIncrementalLinkerCompatible(bool Value) {
    IncrementalLinkerCompatible = Value;
  }

  bool hasLayout() const { return HasLayout; }
  bool getRelaxAll() const { return RelaxAll; }
  void setRelaxAll(bool Value) { RelaxAll = Value; }

  bool isBundlingEnabled() const { return BundleAlignSize != 0; }

  unsigned getBundleAlignSize() const { return BundleAlignSize; }

  void setBundleAlignSize(unsigned Size) {
    assert((Size == 0 || !(Size & (Size - 1))) &&
           "Expect a power-of-two bundle align size");
    BundleAlignSize = Size;
  }

  const_iterator begin() const { return Sections.begin(); }
  const_iterator end() const { return Sections.end(); }

  iterator_range<pointee_iterator<
      typename SmallVector<const MCSymbol *, 0>::const_iterator>>
  symbols() const {
    return make_pointee_range(Symbols);
  }

  /// @}
  /// \name Linker Option List Access
  /// @{

  std::vector<std::vector<std::string>> &getLinkerOptions() {
    return LinkerOptions;
  }

  // FIXME: This is a total hack, this should not be here. Once things are
  // factored so that the streamer has direct access to the .o writer, it can
  // disappear.
  MCLOHContainer &getLOHContainer() { return LOHContainer; }
  const MCLOHContainer &getLOHContainer() const {
    return const_cast<MCAssembler *>(this)->getLOHContainer();
  }

  struct CGProfileEntry {
    const MCSymbolRefExpr *From;
    const MCSymbolRefExpr *To;
    uint64_t Count;
  };
  std::vector<CGProfileEntry> CGProfile;
  /// @}
  /// \name Backend Data Access
  /// @{

  bool registerSection(MCSection &Section);
  bool registerSymbol(const MCSymbol &Symbol);

  MutableArrayRef<std::pair<std::string, size_t>> getFileNames() {
    return FileNames;
  }

  void addFileName(StringRef FileName) {
    FileNames.emplace_back(std::string(FileName), Symbols.size());
  }

  void setCompilerVersion(std::string CompilerVers) {
    if (CompilerVersion.empty())
      CompilerVersion = std::move(CompilerVers);
  }
  StringRef getCompilerVersion() { return CompilerVersion; }

  /// Write the necessary bundle padding to \p OS.
  /// Expects a fragment \p F containing instructions and its size \p FSize.
  void writeFragmentPadding(raw_ostream &OS, const MCEncodedFragment &F,
                            uint64_t FSize) const;

  /// @}

  void dump() const;
};

} // end namespace llvm

#endif // LLVM_MC_MCASSEMBLER_H

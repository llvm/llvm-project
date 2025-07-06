//===- MCELFStreamer.h - MCStreamer ELF Object File Interface ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCELFSTREAMER_H
#define LLVM_MC_MCELFSTREAMER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCObjectStreamer.h"

namespace llvm {

class ELFObjectWriter;
class MCContext;
class MCDataFragment;
class MCFragment;
class MCObjectWriter;
class MCSection;
class MCSubtargetInfo;
class MCSymbol;
class MCSymbolRefExpr;
class MCAsmBackend;
class MCCodeEmitter;
class MCExpr;
class MCInst;

class MCELFStreamer : public MCObjectStreamer {
public:
  MCELFStreamer(MCContext &Context, std::unique_ptr<MCAsmBackend> TAB,
                std::unique_ptr<MCObjectWriter> OW,
                std::unique_ptr<MCCodeEmitter> Emitter);

  ~MCELFStreamer() override = default;

  /// state management
  void reset() override {
    SeenIdent = false;
    MCObjectStreamer::reset();
  }

  ELFObjectWriter &getWriter();

  /// \name MCStreamer Interface
  /// @{

  void initSections(bool NoExecStack, const MCSubtargetInfo &STI) override;
  void changeSection(MCSection *Section, uint32_t Subsection = 0) override;
  void emitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override;
  void emitLabelAtPos(MCSymbol *Symbol, SMLoc Loc, MCDataFragment &F,
                      uint64_t Offset) override;
  void emitWeakReference(MCSymbol *Alias, const MCSymbol *Target) override;
  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;
  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override;

  void emitELFSize(MCSymbol *Symbol, const MCExpr *Value) override;
  void emitELFSymverDirective(const MCSymbol *OriginalSym, StringRef Name,
                              bool KeepOriginalSym) override;

  void emitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                             Align ByteAlignment) override;

  void emitValueImpl(const MCExpr *Value, unsigned Size,
                     SMLoc Loc = SMLoc()) override;

  void emitIdent(StringRef IdentString) override;

  void emitValueToAlignment(Align, int64_t, unsigned, unsigned) override;

  void emitCGProfileEntry(const MCSymbolRefExpr *From,
                          const MCSymbolRefExpr *To, uint64_t Count) override;

  // This is final. Override MCTargetStreamer::finish instead for
  // target-specific code.
  void finishImpl() final;

  void emitBundleAlignMode(Align Alignment) override;
  void emitBundleLock(bool AlignToEnd) override;
  void emitBundleUnlock() override;

  /// ELF object attributes section emission support
  struct AttributeItem {
    // This structure holds all attributes, accounting for their string /
    // numeric value, so we can later emit them in declaration order, keeping
    // all in the same vector.
    enum Types {
      HiddenAttribute = 0,
      NumericAttribute,
      TextAttribute,
      NumericAndTextAttributes
    } Type;
    unsigned Tag;
    unsigned IntValue;
    std::string StringValue;
    AttributeItem(Types Ty, unsigned Tg, unsigned IV, std::string SV)
        : Type(Ty), Tag(Tg), IntValue(IV), StringValue(std::move(SV)) {}
  };

  /// ELF object attributes subsection support
  struct AttributeSubSection {
    bool IsActive;
    StringRef VendorName;
    unsigned IsOptional;
    unsigned ParameterType;
    SmallVector<AttributeItem, 64> Content;
  };

  // Attributes that are added and managed entirely by target.
  SmallVector<AttributeItem, 64> Contents;
  void setAttributeItem(unsigned Attribute, unsigned Value,
                        bool OverwriteExisting);
  void setAttributeItem(unsigned Attribute, StringRef Value,
                        bool OverwriteExisting);
  void setAttributeItems(unsigned Attribute, unsigned IntValue,
                         StringRef StringValue, bool OverwriteExisting);
  void emitAttributesSection(StringRef Vendor, const Twine &Section,
                             unsigned Type, MCSection *&AttributeSection) {
    createAttributesSection(Vendor, Section, Type, AttributeSection, Contents);
  }
  void
  emitAttributesSection(MCSection *&AttributeSection, const Twine &Section,
                        unsigned Type,
                        SmallVector<AttributeSubSection, 64> &SubSectionVec) {
    createAttributesWithSubsection(AttributeSection, Section, Type,
                                   SubSectionVec);
  }

private:
  AttributeItem *getAttributeItem(unsigned Attribute);
  size_t calculateContentSize(SmallVector<AttributeItem, 64> &AttrsVec) const;
  void createAttributesSection(StringRef Vendor, const Twine &Section,
                               unsigned Type, MCSection *&AttributeSection,
                               SmallVector<AttributeItem, 64> &AttrsVec);
  void createAttributesWithSubsection(
      MCSection *&AttributeSection, const Twine &Section, unsigned Type,
      SmallVector<AttributeSubSection, 64> &SubSectionVec);

  // GNU attributes that will get emitted at the end of the asm file.
  SmallVector<AttributeItem, 64> GNUAttributes;

public:
  void emitGNUAttribute(unsigned Tag, unsigned Value) override {
    AttributeItem Item = {AttributeItem::NumericAttribute, Tag, Value,
                          std::string(StringRef(""))};
    GNUAttributes.push_back(Item);
  }

private:
  bool isBundleLocked() const;
  void emitInstToData(const MCInst &Inst, const MCSubtargetInfo &) override;

  void finalizeCGProfileEntry(const MCSymbolRefExpr *&S, uint64_t Offset);
  void finalizeCGProfile();

  bool SeenIdent = false;
};

MCELFStreamer *createARMELFStreamer(MCContext &Context,
                                    std::unique_ptr<MCAsmBackend> TAB,
                                    std::unique_ptr<MCObjectWriter> OW,
                                    std::unique_ptr<MCCodeEmitter> Emitter,
                                    bool IsThumb, bool IsAndroid);

} // end namespace llvm

#endif // LLVM_MC_MCELFSTREAMER_H

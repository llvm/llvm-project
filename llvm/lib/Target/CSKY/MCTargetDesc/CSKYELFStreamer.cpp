//===-- CSKYELFStreamer.cpp - CSKY ELF Target Streamer Methods ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides CSKY specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "CSKYELFStreamer.h"
#include "CSKYMCTargetDesc.h"
#include "MCTargetDesc/CSKYAsmBackend.h"
#include "MCTargetDesc/CSKYBaseInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;

// This part is for ELF object output.
CSKYTargetELFStreamer::CSKYTargetELFStreamer(MCStreamer &S,
                                             const MCSubtargetInfo &STI)
    : CSKYTargetStreamer(S), CurrentVendor("csky") {
  ELFObjectWriter &W = getStreamer().getWriter();
  const FeatureBitset &Features = STI.getFeatureBits();

  unsigned EFlags = W.getELFHeaderEFlags();

  EFlags |= ELF::EF_CSKY_ABIV2;

  if (Features[CSKY::ProcCK801])
    EFlags |= ELF::EF_CSKY_801;
  else if (Features[CSKY::ProcCK802])
    EFlags |= ELF::EF_CSKY_802;
  else if (Features[CSKY::ProcCK803])
    EFlags |= ELF::EF_CSKY_803;
  else if (Features[CSKY::ProcCK804])
    EFlags |= ELF::EF_CSKY_803;
  else if (Features[CSKY::ProcCK805])
    EFlags |= ELF::EF_CSKY_805;
  else if (Features[CSKY::ProcCK807])
    EFlags |= ELF::EF_CSKY_807;
  else if (Features[CSKY::ProcCK810])
    EFlags |= ELF::EF_CSKY_810;
  else if (Features[CSKY::ProcCK860])
    EFlags |= ELF::EF_CSKY_860;
  else
    EFlags |= ELF::EF_CSKY_810;

  if (Features[CSKY::FeatureFPUV2_SF] || Features[CSKY::FeatureFPUV3_SF])
    EFlags |= ELF::EF_CSKY_FLOAT;

  EFlags |= ELF::EF_CSKY_EFV1;

  W.setELFHeaderEFlags(EFlags);
}

MCELFStreamer &CSKYTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

void CSKYTargetELFStreamer::emitAttribute(unsigned Attribute, unsigned Value) {
  setAttributeItem(Attribute, Value, /*OverwriteExisting=*/true);
}

void CSKYTargetELFStreamer::emitTextAttribute(unsigned Attribute,
                                              StringRef String) {
  setAttributeItem(Attribute, String, /*OverwriteExisting=*/true);
}

void CSKYTargetELFStreamer::finishAttributeSection() {
  if (Contents.empty())
    return;

  if (AttributeSection) {
    Streamer.switchSection(AttributeSection);
  } else {
    MCAssembler &MCA = getStreamer().getAssembler();
    AttributeSection = MCA.getContext().getELFSection(
        ".csky.attributes", ELF::SHT_CSKY_ATTRIBUTES, 0);
    Streamer.switchSection(AttributeSection);
    Streamer.emitInt8(ELFAttrs::Format_Version);
  }

  // Vendor size + Vendor name + '\0'
  const size_t VendorHeaderSize = 4 + CurrentVendor.size() + 1;

  // Tag + Tag Size
  const size_t TagHeaderSize = 1 + 4;

  const size_t ContentsSize = calculateContentSize();

  Streamer.emitInt32(VendorHeaderSize + TagHeaderSize + ContentsSize);
  Streamer.emitBytes(CurrentVendor);
  Streamer.emitInt8(0); // '\0'

  Streamer.emitInt8(ELFAttrs::File);
  Streamer.emitInt32(TagHeaderSize + ContentsSize);

  // Size should have been accounted for already, now
  // emit each field as its type (ULEB or String).
  for (AttributeItem item : Contents) {
    Streamer.emitULEB128IntValue(item.Tag);
    switch (item.Type) {
    default:
      llvm_unreachable("Invalid attribute type");
    case AttributeType::Numeric:
      Streamer.emitULEB128IntValue(item.IntValue);
      break;
    case AttributeType::Text:
      Streamer.emitBytes(item.StringValue);
      Streamer.emitInt8(0); // '\0'
      break;
    case AttributeType::NumericAndText:
      Streamer.emitULEB128IntValue(item.IntValue);
      Streamer.emitBytes(item.StringValue);
      Streamer.emitInt8(0); // '\0'
      break;
    }
  }

  Contents.clear();
}

size_t CSKYTargetELFStreamer::calculateContentSize() const {
  size_t Result = 0;
  for (AttributeItem item : Contents) {
    switch (item.Type) {
    case AttributeType::Hidden:
      break;
    case AttributeType::Numeric:
      Result += getULEB128Size(item.Tag);
      Result += getULEB128Size(item.IntValue);
      break;
    case AttributeType::Text:
      Result += getULEB128Size(item.Tag);
      Result += item.StringValue.size() + 1; // string + '\0'
      break;
    case AttributeType::NumericAndText:
      Result += getULEB128Size(item.Tag);
      Result += getULEB128Size(item.IntValue);
      Result += item.StringValue.size() + 1; // string + '\0';
      break;
    }
  }
  return Result;
}

void CSKYELFStreamer::EmitMappingSymbol(StringRef Name) {
  if (Name == "$d" && State == EMS_Data)
    return;
  if (Name == "$t" && State == EMS_Text)
    return;
  if (Name == "$t" && State == EMS_None) {
    State = EMS_Text;
    return;
  }

  State = (Name == "$t" ? EMS_Text : EMS_Data);

  auto *Symbol =
      static_cast<MCSymbolELF *>(getContext().createLocalSymbol(Name));
  emitLabel(Symbol);

  Symbol->setType(ELF::STT_NOTYPE);
  Symbol->setBinding(ELF::STB_LOCAL);
}

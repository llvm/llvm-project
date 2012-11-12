//===--- lib/CodeGen/DIE.cpp - DWARF Info Entries -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Data structures for DWARF info entries.
// 
//===----------------------------------------------------------------------===//

#include "DIE.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/DataLayout.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// DIEAbbrevData Implementation
//===----------------------------------------------------------------------===//

/// Profile - Used to gather unique data for the abbreviation folding set.
///
void DIEAbbrevData::Profile(FoldingSetNodeID &ID) const {
  ID.AddInteger(Attribute);
  ID.AddInteger(Form);
}

//===----------------------------------------------------------------------===//
// DIEAbbrev Implementation
//===----------------------------------------------------------------------===//

/// Profile - Used to gather unique data for the abbreviation folding set.
///
void DIEAbbrev::Profile(FoldingSetNodeID &ID) const {
  ID.AddInteger(Tag);
  ID.AddInteger(ChildrenFlag);

  // For each attribute description.
  for (unsigned i = 0, N = Data.size(); i < N; ++i)
    Data[i].Profile(ID);
}

/// Emit - Print the abbreviation using the specified asm printer.
///
void DIEAbbrev::Emit(AsmPrinter *AP) const {
  // Emit its Dwarf tag type.
  // FIXME: Doing work even in non-asm-verbose runs.
  AP->EmitULEB128(Tag, dwarf::TagString(Tag));

  // Emit whether it has children DIEs.
  // FIXME: Doing work even in non-asm-verbose runs.
  AP->EmitULEB128(ChildrenFlag, dwarf::ChildrenString(ChildrenFlag));

  // For each attribute description.
  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    const DIEAbbrevData &AttrData = Data[i];

    // Emit attribute type.
    // FIXME: Doing work even in non-asm-verbose runs.
    AP->EmitULEB128(AttrData.getAttribute(),
                    dwarf::AttributeString(AttrData.getAttribute()));

    // Emit form type.
    // FIXME: Doing work even in non-asm-verbose runs.
    AP->EmitULEB128(AttrData.getForm(),
                    dwarf::FormEncodingString(AttrData.getForm()));
  }

  // Mark end of abbreviation.
  AP->EmitULEB128(0, "EOM(1)");
  AP->EmitULEB128(0, "EOM(2)");
}

#ifndef NDEBUG
void DIEAbbrev::print(raw_ostream &O) {
  O << "Abbreviation @"
    << format("0x%lx", (long)(intptr_t)this)
    << "  "
    << dwarf::TagString(Tag)
    << " "
    << dwarf::ChildrenString(ChildrenFlag)
    << '\n';

  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    O << "  "
      << dwarf::AttributeString(Data[i].getAttribute())
      << "  "
      << dwarf::FormEncodingString(Data[i].getForm())
      << '\n';
  }
}
void DIEAbbrev::dump() { print(dbgs()); }
#endif

//===----------------------------------------------------------------------===//
// DIE Implementation
//===----------------------------------------------------------------------===//

DIE::~DIE() {
  for (unsigned i = 0, N = Children.size(); i < N; ++i)
    delete Children[i];
}

#ifndef NDEBUG
void DIE::print(raw_ostream &O, unsigned IncIndent) {
  IndentCount += IncIndent;
  const std::string Indent(IndentCount, ' ');
  bool isBlock = Abbrev.getTag() == 0;

  if (!isBlock) {
    O << Indent
      << "Die: "
      << format("0x%lx", (long)(intptr_t)this)
      << ", Offset: " << Offset
      << ", Size: " << Size << "\n";

    O << Indent
      << dwarf::TagString(Abbrev.getTag())
      << " "
      << dwarf::ChildrenString(Abbrev.getChildrenFlag()) << "\n";
  } else {
    O << "Size: " << Size << "\n";
  }

  const SmallVector<DIEAbbrevData, 8> &Data = Abbrev.getData();

  IndentCount += 2;
  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    O << Indent;

    if (!isBlock)
      O << dwarf::AttributeString(Data[i].getAttribute());
    else
      O << "Blk[" << i << "]";

    O <<  "  "
      << dwarf::FormEncodingString(Data[i].getForm())
      << " ";
    Values[i]->print(O);
    O << "\n";
  }
  IndentCount -= 2;

  for (unsigned j = 0, M = Children.size(); j < M; ++j) {
    Children[j]->print(O, 4);
  }

  if (!isBlock) O << "\n";
  IndentCount -= IncIndent;
}

void DIE::dump() {
  print(dbgs());
}
#endif

void DIEValue::anchor() { }

#ifndef NDEBUG
void DIEValue::dump() {
  print(dbgs());
}
#endif

//===----------------------------------------------------------------------===//
// DIEInteger Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit integer of appropriate size.
///
void DIEInteger::EmitValue(AsmPrinter *Asm, unsigned Form) const {
  unsigned Size = ~0U;
  switch (Form) {
  case dwarf::DW_FORM_flag_present:
    // Emit something to keep the lines and comments in sync.
    // FIXME: Is there a better way to do this?
    if (Asm->OutStreamer.hasRawTextSupport())
      Asm->OutStreamer.EmitRawText(StringRef(""));
    return;
  case dwarf::DW_FORM_flag:  // Fall thru
  case dwarf::DW_FORM_ref1:  // Fall thru
  case dwarf::DW_FORM_data1: Size = 1; break;
  case dwarf::DW_FORM_ref2:  // Fall thru
  case dwarf::DW_FORM_data2: Size = 2; break;
  case dwarf::DW_FORM_ref4:  // Fall thru
  case dwarf::DW_FORM_data4: Size = 4; break;
  case dwarf::DW_FORM_ref8:  // Fall thru
  case dwarf::DW_FORM_data8: Size = 8; break;
  case dwarf::DW_FORM_udata: Asm->EmitULEB128(Integer); return;
  case dwarf::DW_FORM_sdata: Asm->EmitSLEB128(Integer); return;
  case dwarf::DW_FORM_addr:
    Size = Asm->getDataLayout().getPointerSize(); break;
  default: llvm_unreachable("DIE Value form not supported yet");
  }
  Asm->OutStreamer.EmitIntValue(Integer, Size, 0/*addrspace*/);
}

/// SizeOf - Determine size of integer value in bytes.
///
unsigned DIEInteger::SizeOf(AsmPrinter *AP, unsigned Form) const {
  switch (Form) {
  case dwarf::DW_FORM_flag_present: return 0;
  case dwarf::DW_FORM_flag:  // Fall thru
  case dwarf::DW_FORM_ref1:  // Fall thru
  case dwarf::DW_FORM_data1: return sizeof(int8_t);
  case dwarf::DW_FORM_ref2:  // Fall thru
  case dwarf::DW_FORM_data2: return sizeof(int16_t);
  case dwarf::DW_FORM_ref4:  // Fall thru
  case dwarf::DW_FORM_data4: return sizeof(int32_t);
  case dwarf::DW_FORM_ref8:  // Fall thru
  case dwarf::DW_FORM_data8: return sizeof(int64_t);
  case dwarf::DW_FORM_udata: return MCAsmInfo::getULEB128Size(Integer);
  case dwarf::DW_FORM_sdata: return MCAsmInfo::getSLEB128Size(Integer);
  case dwarf::DW_FORM_addr:  return AP->getDataLayout().getPointerSize();
  default: llvm_unreachable("DIE Value form not supported yet");
  }
}

#ifndef NDEBUG
void DIEInteger::print(raw_ostream &O) {
  O << "Int: " << (int64_t)Integer << "  0x";
  O.write_hex(Integer);
}
#endif

//===----------------------------------------------------------------------===//
// DIELabel Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit label value.
///
void DIELabel::EmitValue(AsmPrinter *AP, unsigned Form) const {
  AP->OutStreamer.EmitSymbolValue(Label, SizeOf(AP, Form), 0/*AddrSpace*/);
}

/// SizeOf - Determine size of label value in bytes.
///
unsigned DIELabel::SizeOf(AsmPrinter *AP, unsigned Form) const {
  if (Form == dwarf::DW_FORM_data4) return 4;
  if (Form == dwarf::DW_FORM_strp) return 4;
  return AP->getDataLayout().getPointerSize();
}

#ifndef NDEBUG
void DIELabel::print(raw_ostream &O) {
  O << "Lbl: " << Label->getName();
}
#endif

//===----------------------------------------------------------------------===//
// DIEDelta Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit delta value.
///
void DIEDelta::EmitValue(AsmPrinter *AP, unsigned Form) const {
  AP->EmitLabelDifference(LabelHi, LabelLo, SizeOf(AP, Form));
}

/// SizeOf - Determine size of delta value in bytes.
///
unsigned DIEDelta::SizeOf(AsmPrinter *AP, unsigned Form) const {
  if (Form == dwarf::DW_FORM_data4) return 4;
  if (Form == dwarf::DW_FORM_strp) return 4;
  return AP->getDataLayout().getPointerSize();
}

#ifndef NDEBUG
void DIEDelta::print(raw_ostream &O) {
  O << "Del: " << LabelHi->getName() << "-" << LabelLo->getName();
}
#endif

//===----------------------------------------------------------------------===//
// DIEEntry Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit debug information entry offset.
///
void DIEEntry::EmitValue(AsmPrinter *AP, unsigned Form) const {
  AP->EmitInt32(Entry->getOffset());
}

#ifndef NDEBUG
void DIEEntry::print(raw_ostream &O) {
  O << format("Die: 0x%lx", (long)(intptr_t)Entry);
}
#endif

//===----------------------------------------------------------------------===//
// DIEBlock Implementation
//===----------------------------------------------------------------------===//

/// ComputeSize - calculate the size of the block.
///
unsigned DIEBlock::ComputeSize(AsmPrinter *AP) {
  if (!Size) {
    const SmallVector<DIEAbbrevData, 8> &AbbrevData = Abbrev.getData();
    for (unsigned i = 0, N = Values.size(); i < N; ++i)
      Size += Values[i]->SizeOf(AP, AbbrevData[i].getForm());
  }

  return Size;
}

/// EmitValue - Emit block data.
///
void DIEBlock::EmitValue(AsmPrinter *Asm, unsigned Form) const {
  switch (Form) {
  default: llvm_unreachable("Improper form for block");
  case dwarf::DW_FORM_block1: Asm->EmitInt8(Size);    break;
  case dwarf::DW_FORM_block2: Asm->EmitInt16(Size);   break;
  case dwarf::DW_FORM_block4: Asm->EmitInt32(Size);   break;
  case dwarf::DW_FORM_block:  Asm->EmitULEB128(Size); break;
  }

  const SmallVector<DIEAbbrevData, 8> &AbbrevData = Abbrev.getData();
  for (unsigned i = 0, N = Values.size(); i < N; ++i)
    Values[i]->EmitValue(Asm, AbbrevData[i].getForm());
}

/// SizeOf - Determine size of block data in bytes.
///
unsigned DIEBlock::SizeOf(AsmPrinter *AP, unsigned Form) const {
  switch (Form) {
  case dwarf::DW_FORM_block1: return Size + sizeof(int8_t);
  case dwarf::DW_FORM_block2: return Size + sizeof(int16_t);
  case dwarf::DW_FORM_block4: return Size + sizeof(int32_t);
  case dwarf::DW_FORM_block:  return Size + MCAsmInfo::getULEB128Size(Size);
  default: llvm_unreachable("Improper form for block");
  }
}

#ifndef NDEBUG
void DIEBlock::print(raw_ostream &O) {
  O << "Blk: ";
  DIE::print(O, 5);
}
#endif

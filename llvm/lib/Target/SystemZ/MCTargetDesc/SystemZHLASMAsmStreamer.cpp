//===- SystemZHLASMAsmStreamer.cpp - HLASM Assembly Text Output -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZHLASMAsmStreamer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCGOFFAttributes.h"
#include "llvm/MC/MCGOFFStreamer.h"
#include "llvm/MC/MCSymbolGOFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Signals.h"
#include <sstream>

using namespace llvm;

void SystemZHLASMAsmStreamer::visitUsedSymbol(const MCSymbol &Sym) {
  Assembler->registerSymbol(Sym);
}

void SystemZHLASMAsmStreamer::EmitEOL() {
  // Comments are emitted on a new line before the instruction.
  if (IsVerboseAsm)
    EmitComment();

  std::istringstream Stream(Str);
  SmallVector<std::string> Lines;
  std::string Line;
  while (std::getline(Stream, Line, '\n'))
    Lines.push_back(Line);

  for (auto S : Lines) {
    if (LLVM_LIKELY(S.length() < ContIndicatorColumn)) {
      FOS << S;
      // Each line in HLASM must fill the full 80 characters.
      FOS.PadToColumn(InstLimit);
      FOS << "\n";
    } else {
      // If last character before end of the line is not a space
      // we must insert an additional non-space character that
      // is not part of the statement coding. We just reuse
      // the existing character by making the new substring start
      // 1 character sooner, thus "duplicating" that character
      // If The last character is a space. We insert an X instead.
      std::string TmpSubStr = S.substr(0, ContIndicatorColumn);
      if (!TmpSubStr.compare(ContIndicatorColumn - 1, 1, " "))
        TmpSubStr.replace(ContIndicatorColumn - 1, 1, "X");

      FOS << TmpSubStr;
      FOS.PadToColumn(InstLimit);
      FOS << "\n";

      size_t Emitted = ContIndicatorColumn - 1;

      while (Emitted < S.length()) {
        if ((S.length() - Emitted) < ContLen)
          TmpSubStr = S.substr(Emitted, S.length());
        else {
          TmpSubStr = S.substr(Emitted, ContLen);
          if (!TmpSubStr.compare(ContLen - 1, 1, " "))
            TmpSubStr.replace(ContLen - 1, 1, "X");
        }
        FOS.PadToColumn(ContStartColumn);
        FOS << TmpSubStr;
        FOS.PadToColumn(InstLimit);
        FOS << "\n";
        Emitted += ContLen - 1;
      }
    }
  }
  Str.clear();
}

void SystemZHLASMAsmStreamer::changeSection(MCSection *Section,
                                            uint32_t Subsection) {
  MAI->printSwitchToSection(*Section, Subsection,
                            getContext().getTargetTriple(), OS);
  MCStreamer::changeSection(Section, Subsection);
  EmitEOL();
}

void SystemZHLASMAsmStreamer::emitAlignmentDS(uint64_t ByteAlignment,
                                              std::optional<int64_t> Value,
                                              unsigned ValueSize,
                                              unsigned MaxBytesToEmit) {
  if (!isPowerOf2_64(ByteAlignment))
    report_fatal_error("Only power-of-two alignments are supported ");

  OS << " DS 0";
  switch (ValueSize) {
  default:
    llvm_unreachable("Invalid size for machine code value!");
  case 1:
    OS << "B";
    break;
  case 2:
    OS << "H";
    break;
  case 4:
    OS << "F";
    break;
  case 8:
    OS << "D";
    break;
  case 16:
    OS << "Q";
    break;
  }

  EmitEOL();
}

void SystemZHLASMAsmStreamer::AddComment(const Twine &T, bool EOL) {
  if (!IsVerboseAsm)
    return;

  T.toVector(CommentToEmit);

  if (EOL)
    CommentToEmit.push_back('\n'); // Place comment in a new line.
}

void SystemZHLASMAsmStreamer::EmitComment() {
  if (CommentToEmit.empty() && CommentStream.GetNumBytesInBuffer() == 0)
    return;

  StringRef Comments = CommentToEmit;

  assert(Comments.back() == '\n' && "Comment array not newline terminated");
  do {
    // Emit a line of comments, but not exceeding 80 characters.
    size_t Position = std::min(InstLimit - 2, Comments.find('\n'));
    FOS << MAI->getCommentString() << ' ' << Comments.substr(0, Position)
        << '\n';

    if (Comments[Position] == '\n')
      Position++;
    Comments = Comments.substr(Position);
  } while (!Comments.empty());

  CommentToEmit.clear();
}

void SystemZHLASMAsmStreamer::emitValueToAlignment(Align Alignment,
                                                   int64_t Fill,
                                                   uint8_t FillLen,
                                                   unsigned MaxBytesToEmit) {
  emitAlignmentDS(Alignment.value(), Fill, FillLen, MaxBytesToEmit);
}

void SystemZHLASMAsmStreamer::emitCodeAlignment(Align Alignment,
                                                const MCSubtargetInfo *STI,
                                                unsigned MaxBytesToEmit) {
  // Emit with a text fill value.
  if (MAI->getTextAlignFillValue())
    emitAlignmentDS(Alignment.value(), MAI->getTextAlignFillValue(), 1,
                    MaxBytesToEmit);
  else
    emitAlignmentDS(Alignment.value(), std::nullopt, 1, MaxBytesToEmit);
}

void SystemZHLASMAsmStreamer::emitBytes(StringRef Data) {
  assert(getCurrentSectionOnly() &&
         "Cannot emit contents before setting section!");
  if (Data.empty())
    return;

  OS << " DC ";
  size_t Len = Data.size();
  SmallVector<uint8_t> Chars;
  Chars.resize(Len);
  OS << "XL" << Len;
  uint32_t Index = 0;
  for (uint8_t C : Data) {
    Chars[Index] = C;
    Index++;
  }

  OS << '\'' << toHex(Chars) << '\'';

  EmitEOL();
}

void SystemZHLASMAsmStreamer::emitInstruction(const MCInst &Inst,
                                              const MCSubtargetInfo &STI) {

  InstPrinter->printInst(&Inst, 0, "", STI, OS);
  EmitEOL();
}

static void emitXATTR(raw_ostream &OS, StringRef Name,
                      GOFF::ESDLinkageType Linkage,
                      GOFF::ESDExecutable Executable,
                      GOFF::ESDBindingScope BindingScope) {
  llvm::ListSeparator Sep(",");
  OS << Name << " XATTR ";
  OS << Sep << "LINKAGE(" << (Linkage == GOFF::ESD_LT_OS ? "OS" : "XPLINK")
     << ")";
  if (Executable != GOFF::ESD_EXE_Unspecified)
    OS << Sep << "REFERENCE("
       << (Executable == GOFF::ESD_EXE_CODE ? "CODE" : "DATA") << ")";
  if (BindingScope != GOFF::ESD_BSC_Unspecified) {
    OS << Sep << "SCOPE(";
    switch (BindingScope) {
    case GOFF::ESD_BSC_Section:
      OS << "SECTION";
      break;
    case GOFF::ESD_BSC_Module:
      OS << "MODULE";
      break;
    case GOFF::ESD_BSC_Library:
      OS << "LIBRARY";
      break;
    case GOFF::ESD_BSC_ImportExport:
      OS << "EXPORT";
      break;
    default:
      break;
    }
    OS << ')';
  }
  OS << '\n';
}

void SystemZHLASMAsmStreamer::emitLabel(MCSymbol *Symbol, SMLoc Loc) {
  MCSymbolGOFF *Sym = static_cast<MCSymbolGOFF *>(Symbol);

  MCStreamer::emitLabel(Sym, Loc);

  // Emit label and ENTRY statement only if not implied by CSECT. Do not emit a
  // label if the symbol is on a PR section.
  bool EmitLabelAndEntry =
      !static_cast<MCSectionGOFF *>(getCurrentSectionOnly())->isPR();
  if (!Sym->isTemporary() && Sym->isInEDSection()) {
    EmitLabelAndEntry =
        Sym->getName() !=
        static_cast<MCSectionGOFF &>(Sym->getSection()).getParent()->getName();
    if (EmitLabelAndEntry) {
      OS << " ENTRY " << Sym->getName();
      EmitEOL();
    }

    emitXATTR(OS, Sym->getName(), Sym->getLinkage(), Sym->getCodeData(),
              Sym->getBindingScope());
    EmitEOL();
  }

  if (EmitLabelAndEntry) {
    OS << Sym->getName() << " DS 0H";
    EmitEOL();
  }
}

bool SystemZHLASMAsmStreamer::emitSymbolAttribute(MCSymbol *Sym,
                                                  MCSymbolAttr Attribute) {
  return static_cast<MCSymbolGOFF *>(Sym)->setSymbolAttribute(Attribute);
}

void SystemZHLASMAsmStreamer::emitRawTextImpl(StringRef String) {
  String.consume_back("\n");
  OS << String;
  EmitEOL();
}

// Slight duplicate of MCExpr::print due to HLASM only recognizing limited
// arithmetic operators (+-*/).
void SystemZHLASMAsmStreamer::emitHLASMValueImpl(const MCExpr *Value,
                                                 unsigned Size, bool Parens) {
  switch (Value->getKind()) {
  case MCExpr::Constant: {
    OS << "XL" << Size << '\'';
    MAI->printExpr(OS, *Value);
    OS << '\'';
    return;
  }
  case MCExpr::Binary: {
    const MCBinaryExpr &BE = cast<MCBinaryExpr>(*Value);
    int64_t Const;
    // Or is handled differently.
    if (BE.getOpcode() == MCBinaryExpr::Or) {
      emitHLASMValueImpl(BE.getLHS(), Size, true);
      OS << ',';
      emitHLASMValueImpl(BE.getRHS(), Size, true);
      return;
    }

    if (Parens)
      OS << "AD(";
    emitHLASMValueImpl(BE.getLHS(), Size);

    switch (BE.getOpcode()) {
    case MCBinaryExpr::LShr: {
      Const = cast<MCConstantExpr>(BE.getRHS())->getValue();
      OS << '/' << (1 << Const);
      if (Parens)
        OS << ')';
      return;
    }
    case MCBinaryExpr::Add:
      OS << '+';
      break;
    case MCBinaryExpr::Div:
      OS << '/';
      break;
    case MCBinaryExpr::Mul:
      OS << '*';
      break;
    case MCBinaryExpr::Sub:
      OS << '-';
      break;
    default:
      getContext().reportError(SMLoc(),
                               "Unrecognized HLASM arithmetic expression!");
    }
    emitHLASMValueImpl(BE.getRHS(), Size);
    if (Parens)
      OS << ')';
    return;
  }
  case MCExpr::Target:
    MAI->printExpr(OS, *Value);
    return;
  default:
    Parens &= isa<MCSymbolRefExpr>(Value);
    if (Parens)
      OS << "AD(";
    MAI->printExpr(OS, *Value);
    if (Parens)
      OS << ')';
    return;
  }
}

void SystemZHLASMAsmStreamer::emitValueImpl(const MCExpr *Value, unsigned Size,
                                            SMLoc Loc) {
  assert(Size <= 8 && "Invalid size");
  assert(getCurrentSectionOnly() &&
         "Cannot emit contents before setting section!");

  MCStreamer::emitValueImpl(Value, Size, Loc);
  OS << " DC ";
  emitHLASMValueImpl(Value, Size, true);
  EmitEOL();
}

void SystemZHLASMAsmStreamer::finishImpl() {
  for (auto &Symbol : getAssembler().symbols()) {
    if (Symbol.isTemporary() || !Symbol.isRegistered() || Symbol.isDefined())
      continue;
    auto &Sym = static_cast<MCSymbolGOFF &>(const_cast<MCSymbol &>(Symbol));
    OS << " " << (Sym.isWeak() ? "WXTRN" : "EXTRN") << " " << Sym.getName();
    EmitEOL();
    emitXATTR(OS, Sym.getName(), Sym.getLinkage(), Sym.getCodeData(),
              Sym.getBindingScope());
    EmitEOL();
  }

  // Finish the assembly output.
  OS << " END";
  EmitEOL();
}

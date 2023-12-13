//===-- SourcePrinter.cpp -  source interleaving utilities ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveVariablePrinter and SourcePrinter classes to
// keep track of DWARF info as the current address is updated, and print out the
// source file line and variable liveness as needed.
//
//===----------------------------------------------------------------------===//

#include "SourcePrinter.h"
#include "llvm-objdump.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "objdump"

namespace llvm {
namespace objdump {

bool LiveVariable::liveAtAddress(object::SectionedAddress Addr) {
  if (LocExpr.Range == std::nullopt)
    return false;
  return LocExpr.Range->SectionIndex == Addr.SectionIndex &&
         LocExpr.Range->LowPC <= Addr.Address &&
         LocExpr.Range->HighPC > Addr.Address;
}

void LiveVariable::print(raw_ostream &OS, const MCRegisterInfo &MRI) const {
  DataExtractor Data({LocExpr.Expr.data(), LocExpr.Expr.size()},
                     Unit->getContext().isLittleEndian(), 0);
  DWARFExpression Expression(Data, Unit->getAddressByteSize());

  auto GetRegName = [&MRI, &OS](uint64_t DwarfRegNum, bool IsEH) -> StringRef {
    if (std::optional<unsigned> LLVMRegNum =
            MRI.getLLVMRegNum(DwarfRegNum, IsEH))
      if (const char *RegName = MRI.getName(*LLVMRegNum))
        return StringRef(RegName);
    OS << "<unknown register " << DwarfRegNum << ">";
    return {};
  };

  Expression.printCompact(OS, GetRegName);
}

void LiveVariablePrinter::addVariable(DWARFDie FuncDie, DWARFDie VarDie) {
  uint64_t FuncLowPC, FuncHighPC, SectionIndex;
  FuncDie.getLowAndHighPC(FuncLowPC, FuncHighPC, SectionIndex);
  const char *VarName = VarDie.getName(DINameKind::ShortName);
  DWARFUnit *U = VarDie.getDwarfUnit();

  Expected<DWARFLocationExpressionsVector> Locs =
      VarDie.getLocations(dwarf::DW_AT_location);
  if (!Locs) {
    // If the variable doesn't have any locations, just ignore it. We don't
    // report an error or warning here as that could be noisy on optimised
    // code.
    consumeError(Locs.takeError());
    return;
  }

  for (const DWARFLocationExpression &LocExpr : *Locs) {
    if (LocExpr.Range) {
      LiveVariables.emplace_back(LocExpr, VarName, U, FuncDie);
    } else {
      // If the LocExpr does not have an associated range, it is valid for
      // the whole of the function.
      // TODO: technically it is not valid for any range covered by another
      // LocExpr, does that happen in reality?
      DWARFLocationExpression WholeFuncExpr{
          DWARFAddressRange(FuncLowPC, FuncHighPC, SectionIndex), LocExpr.Expr};
      LiveVariables.emplace_back(WholeFuncExpr, VarName, U, FuncDie);
    }
  }
}

void LiveVariablePrinter::addFunction(DWARFDie D) {
  for (const DWARFDie &Child : D.children()) {
    if (Child.getTag() == dwarf::DW_TAG_variable ||
        Child.getTag() == dwarf::DW_TAG_formal_parameter)
      addVariable(D, Child);
    else
      addFunction(Child);
  }
}

// Get the column number (in characters) at which the first live variable
// line should be printed.
unsigned LiveVariablePrinter::getIndentLevel() const {
  return GetColumnIndent(STI, DisassemblyColumn::Variables);
}

// Indent to the first live-range column to the right of the currently
// printed line, and return the index of that column.
// TODO: formatted_raw_ostream uses "column" to mean a number of characters
// since the last \n, and we use it to mean the number of slots in which we
// put live variable lines. Pick a less overloaded word.
unsigned LiveVariablePrinter::moveToFirstVarColumn(formatted_raw_ostream &OS) {
  // Logical column number: column zero is the first column we print in, each
  // logical column is 2 physical columns wide.
  unsigned FirstUnprintedLogicalColumn =
      std::max((int)(OS.getColumn() - getIndentLevel() + 1) / 2, 0);
  // Physical column number: the actual column number in characters, with
  // zero being the left-most side of the screen.
  unsigned FirstUnprintedPhysicalColumn =
      getIndentLevel() + FirstUnprintedLogicalColumn * 2;

  if (FirstUnprintedPhysicalColumn > OS.getColumn())
    OS.PadToColumn(FirstUnprintedPhysicalColumn);

  return FirstUnprintedLogicalColumn;
}

unsigned LiveVariablePrinter::findFreeColumn() {
  for (unsigned ColIdx = 0; ColIdx < ActiveCols.size(); ++ColIdx)
    if (!ActiveCols[ColIdx].isActive())
      return ColIdx;

  size_t OldSize = ActiveCols.size();
  ActiveCols.grow(std::max<size_t>(OldSize * 2, 1));
  return OldSize;
}

void LiveVariablePrinter::dump() const {
  for (const LiveVariable &LV : LiveVariables) {
    dbgs() << LV.VarName << " @ " << LV.LocExpr.Range << ": ";
    LV.print(dbgs(), MRI);
    dbgs() << "\n";
  }
}

void LiveVariablePrinter::addCompileUnit(DWARFDie D) {
  if (D.getTag() == dwarf::DW_TAG_subprogram)
    addFunction(D);
  else
    for (const DWARFDie &Child : D.children())
      addFunction(Child);
}

/// Update to match the state of the instruction between ThisAddr and
/// NextAddr. In the common case, any live range active at ThisAddr is
/// live-in to the instruction, and any live range active at NextAddr is
/// live-out of the instruction. If IncludeDefinedVars is false, then live
/// ranges starting at NextAddr will be ignored.
void LiveVariablePrinter::update(object::SectionedAddress ThisAddr,
                                 object::SectionedAddress NextAddr,
                                 bool IncludeDefinedVars) {
  // First, check variables which have already been assigned a column, so
  // that we don't change their order.
  SmallSet<unsigned, 8> CheckedVarIdxs;
  for (unsigned ColIdx = 0, End = ActiveCols.size(); ColIdx < End; ++ColIdx) {
    if (!ActiveCols[ColIdx].isActive())
      continue;
    CheckedVarIdxs.insert(ActiveCols[ColIdx].VarIdx);
    LiveVariable &LV = LiveVariables[ActiveCols[ColIdx].VarIdx];
    ActiveCols[ColIdx].LiveIn = LV.liveAtAddress(ThisAddr);
    ActiveCols[ColIdx].LiveOut = LV.liveAtAddress(NextAddr);
    LLVM_DEBUG(dbgs() << "pass 1, " << ThisAddr.Address << "-"
                      << NextAddr.Address << ", " << LV.VarName << ", Col "
                      << ColIdx << ": LiveIn=" << ActiveCols[ColIdx].LiveIn
                      << ", LiveOut=" << ActiveCols[ColIdx].LiveOut << "\n");

    if (!ActiveCols[ColIdx].LiveIn && !ActiveCols[ColIdx].LiveOut)
      ActiveCols[ColIdx].VarIdx = Column::NullVarIdx;
  }

  // Next, look for variables which don't already have a column, but which
  // are now live.
  if (IncludeDefinedVars) {
    for (unsigned VarIdx = 0, End = LiveVariables.size(); VarIdx < End;
         ++VarIdx) {
      if (CheckedVarIdxs.count(VarIdx))
        continue;
      LiveVariable &LV = LiveVariables[VarIdx];
      bool LiveIn = LV.liveAtAddress(ThisAddr);
      bool LiveOut = LV.liveAtAddress(NextAddr);
      if (!LiveIn && !LiveOut)
        continue;

      unsigned ColIdx = findFreeColumn();
      LLVM_DEBUG(dbgs() << "pass 2, " << ThisAddr.Address << "-"
                        << NextAddr.Address << ", " << LV.VarName << ", Col "
                        << ColIdx << ": LiveIn=" << LiveIn
                        << ", LiveOut=" << LiveOut << "\n");
      ActiveCols[ColIdx].VarIdx = VarIdx;
      ActiveCols[ColIdx].LiveIn = LiveIn;
      ActiveCols[ColIdx].LiveOut = LiveOut;
      ActiveCols[ColIdx].MustDrawLabel = true;
    }
  }
}

enum class LineChar {
  RangeStart,
  RangeMid,
  RangeEnd,
  LabelVert,
  LabelCornerNew,
  LabelCornerActive,
  LabelHoriz,
};
const char *LiveVariablePrinter::getLineChar(LineChar C) const {
  bool IsASCII = DbgVariables == DVASCII;
  switch (C) {
  case LineChar::RangeStart:
    return IsASCII ? "^" : u8"\u2548";
  case LineChar::RangeMid:
    return IsASCII ? "|" : u8"\u2503";
  case LineChar::RangeEnd:
    return IsASCII ? "v" : u8"\u253b";
  case LineChar::LabelVert:
    return IsASCII ? "|" : u8"\u2502";
  case LineChar::LabelCornerNew:
    return IsASCII ? "/" : u8"\u250c";
  case LineChar::LabelCornerActive:
    return IsASCII ? "|" : u8"\u2520";
  case LineChar::LabelHoriz:
    return IsASCII ? "-" : u8"\u2500";
  }
  llvm_unreachable("Unhandled LineChar enum");
}

/// Print live ranges to the right of an existing line. This assumes the
/// line is not an instruction, so doesn't start or end any live ranges, so
/// we only need to print active ranges or empty columns. If AfterInst is
/// true, this is being printed after the last instruction fed to update(),
/// otherwise this is being printed before it.
void LiveVariablePrinter::printAfterOtherLine(formatted_raw_ostream &OS,
                                              bool AfterInst) {
  if (ActiveCols.size()) {
    unsigned FirstUnprintedColumn = moveToFirstVarColumn(OS);
    for (size_t ColIdx = FirstUnprintedColumn, End = ActiveCols.size();
         ColIdx < End; ++ColIdx) {
      if (ActiveCols[ColIdx].isActive()) {
        if ((AfterInst && ActiveCols[ColIdx].LiveOut) ||
            (!AfterInst && ActiveCols[ColIdx].LiveIn))
          OS << getLineChar(LineChar::RangeMid);
        else if (!AfterInst && ActiveCols[ColIdx].LiveOut)
          OS << getLineChar(LineChar::LabelVert);
        else
          OS << " ";
      }
      OS << " ";
    }
  }
  OS << "\n";
}

/// Print any live variable range info needed to the right of a
/// non-instruction line of disassembly. This is where we print the variable
/// names and expressions, with thin line-drawing characters connecting them
/// to the live range which starts at the next instruction. If MustPrint is
/// true, we have to print at least one line (with the continuation of any
/// already-active live ranges) because something has already been printed
/// earlier on this line.
void LiveVariablePrinter::printBetweenInsts(formatted_raw_ostream &OS,
                                            bool MustPrint, uint64_t Addr,
                                            ControlFlowPrinter *CFP) {
  bool PrintedSomething = false;
  for (unsigned ColIdx = 0, End = ActiveCols.size(); ColIdx < End; ++ColIdx) {
    if (ActiveCols[ColIdx].isActive() && ActiveCols[ColIdx].MustDrawLabel) {
      // First we need to print the live range markers for any active
      // columns to the left of this one.
      if (CFP)
        CFP->printOther(OS, Addr);
      OS.PadToColumn(getIndentLevel());
      for (unsigned ColIdx2 = 0; ColIdx2 < ColIdx; ++ColIdx2) {
        if (ActiveCols[ColIdx2].isActive()) {
          if (ActiveCols[ColIdx2].MustDrawLabel && !ActiveCols[ColIdx2].LiveIn)
            OS << getLineChar(LineChar::LabelVert) << " ";
          else
            OS << getLineChar(LineChar::RangeMid) << " ";
        } else
          OS << "  ";
      }

      // Then print the variable name and location of the new live range,
      // with box drawing characters joining it to the live range line.
      OS << getLineChar(ActiveCols[ColIdx].LiveIn ? LineChar::LabelCornerActive
                                                  : LineChar::LabelCornerNew)
         << getLineChar(LineChar::LabelHoriz) << " ";
      WithColor(OS, raw_ostream::GREEN)
          << LiveVariables[ActiveCols[ColIdx].VarIdx].VarName;
      OS << " = ";
      {
        WithColor ExprColor(OS, raw_ostream::CYAN);
        LiveVariables[ActiveCols[ColIdx].VarIdx].print(OS, MRI);
      }

      // If there are any columns to the right of the expression we just
      // printed, then continue their live range lines.
      unsigned FirstUnprintedColumn = moveToFirstVarColumn(OS);
      for (unsigned ColIdx2 = FirstUnprintedColumn, End = ActiveCols.size();
           ColIdx2 < End; ++ColIdx2) {
        if (ActiveCols[ColIdx2].isActive() && ActiveCols[ColIdx2].LiveIn)
          OS << getLineChar(LineChar::RangeMid) << " ";
        else
          OS << "  ";
      }

      OS << "\n";
      PrintedSomething = true;
    }
  }

  for (unsigned ColIdx = 0, End = ActiveCols.size(); ColIdx < End; ++ColIdx)
    if (ActiveCols[ColIdx].isActive())
      ActiveCols[ColIdx].MustDrawLabel = false;

  // If we must print something (because we printed a line/column number),
  // but don't have any new variables to print, then print a line which
  // just continues any existing live ranges.
  if (MustPrint && !PrintedSomething)
    printAfterOtherLine(OS, false);
}

/// Print the live variable ranges to the right of a disassembled instruction.
void LiveVariablePrinter::printAfterInst(formatted_raw_ostream &OS) {
  if (!ActiveCols.size())
    return;
  unsigned FirstUnprintedColumn = moveToFirstVarColumn(OS);
  for (unsigned ColIdx = FirstUnprintedColumn, End = ActiveCols.size();
       ColIdx < End; ++ColIdx) {
    if (!ActiveCols[ColIdx].isActive())
      OS << "  ";
    else if (ActiveCols[ColIdx].LiveIn && ActiveCols[ColIdx].LiveOut)
      OS << getLineChar(LineChar::RangeMid) << " ";
    else if (ActiveCols[ColIdx].LiveOut)
      OS << getLineChar(LineChar::RangeStart) << " ";
    else if (ActiveCols[ColIdx].LiveIn)
      OS << getLineChar(LineChar::RangeEnd) << " ";
    else
      llvm_unreachable("var must be live in or out!");
  }
}

bool SourcePrinter::cacheSource(const DILineInfo &LineInfo) {
  std::unique_ptr<MemoryBuffer> Buffer;
  if (LineInfo.Source) {
    Buffer = MemoryBuffer::getMemBuffer(*LineInfo.Source);
  } else {
    auto BufferOrError = MemoryBuffer::getFile(LineInfo.FileName);
    if (!BufferOrError) {
      if (MissingSources.insert(LineInfo.FileName).second)
        reportWarning("failed to find source " + LineInfo.FileName,
                      Obj->getFileName());
      return false;
    }
    Buffer = std::move(*BufferOrError);
  }
  // Chomp the file to get lines
  const char *BufferStart = Buffer->getBufferStart(),
             *BufferEnd = Buffer->getBufferEnd();
  std::vector<StringRef> &Lines = LineCache[LineInfo.FileName];
  const char *Start = BufferStart;
  for (const char *I = BufferStart; I != BufferEnd; ++I)
    if (*I == '\n') {
      Lines.emplace_back(Start, I - Start - (BufferStart < I && I[-1] == '\r'));
      Start = I + 1;
    }
  if (Start < BufferEnd)
    Lines.emplace_back(Start, BufferEnd - Start);
  SourceCache[LineInfo.FileName] = std::move(Buffer);
  return true;
}

void SourcePrinter::printSourceLine(formatted_raw_ostream &OS,
                                    object::SectionedAddress Address,
                                    StringRef ObjectFilename,
                                    LiveVariablePrinter &LVP,
                                    StringRef Delimiter) {
  if (!Symbolizer)
    return;

  DILineInfo LineInfo = DILineInfo();
  Expected<DILineInfo> ExpectedLineInfo =
      Symbolizer->symbolizeCode(*Obj, Address);
  std::string ErrorMessage;
  if (ExpectedLineInfo) {
    LineInfo = *ExpectedLineInfo;
  } else if (!WarnedInvalidDebugInfo) {
    WarnedInvalidDebugInfo = true;
    // TODO Untested.
    reportWarning("failed to parse debug information: " +
                      toString(ExpectedLineInfo.takeError()),
                  ObjectFilename);
  }

  if (!objdump::Prefix.empty() &&
      sys::path::is_absolute_gnu(LineInfo.FileName)) {
    // FileName has at least one character since is_absolute_gnu is false for
    // an empty string.
    assert(!LineInfo.FileName.empty());
    if (PrefixStrip > 0) {
      uint32_t Level = 0;
      auto StrippedNameStart = LineInfo.FileName.begin();

      // Path.h iterator skips extra separators. Therefore it cannot be used
      // here to keep compatibility with GNU Objdump.
      for (auto Pos = StrippedNameStart + 1, End = LineInfo.FileName.end();
           Pos != End && Level < PrefixStrip; ++Pos) {
        if (sys::path::is_separator(*Pos)) {
          StrippedNameStart = Pos;
          ++Level;
        }
      }

      LineInfo.FileName =
          std::string(StrippedNameStart, LineInfo.FileName.end());
    }

    SmallString<128> FilePath;
    sys::path::append(FilePath, Prefix, LineInfo.FileName);

    LineInfo.FileName = std::string(FilePath);
  }

  if (PrintLines)
    printLines(OS, LineInfo, Delimiter, LVP);
  if (PrintSource)
    printSources(OS, LineInfo, ObjectFilename, Delimiter, LVP);
  OldLineInfo = LineInfo;
}

void SourcePrinter::printLines(formatted_raw_ostream &OS,
                               const DILineInfo &LineInfo, StringRef Delimiter,
                               LiveVariablePrinter &LVP) {
  bool PrintFunctionName = LineInfo.FunctionName != DILineInfo::BadString &&
                           LineInfo.FunctionName != OldLineInfo.FunctionName;
  if (PrintFunctionName) {
    OS << Delimiter << LineInfo.FunctionName;
    // If demangling is successful, FunctionName will end with "()". Print it
    // only if demangling did not run or was unsuccessful.
    if (!StringRef(LineInfo.FunctionName).ends_with("()"))
      OS << "()";
    OS << ":\n";
  }
  if (LineInfo.FileName != DILineInfo::BadString && LineInfo.Line != 0 &&
      (OldLineInfo.Line != LineInfo.Line ||
       OldLineInfo.FileName != LineInfo.FileName || PrintFunctionName)) {
    OS << Delimiter << LineInfo.FileName << ":" << LineInfo.Line;
    LVP.printBetweenInsts(OS, true);
  }
}

// Get the source line text for LineInfo:
// - use LineInfo::LineSource if available;
// - use LineCache if LineInfo::Source otherwise.
StringRef SourcePrinter::getLine(const DILineInfo &LineInfo,
                                 StringRef ObjectFilename) {
  if (LineInfo.LineSource)
    return LineInfo.LineSource.value();

  if (SourceCache.find(LineInfo.FileName) == SourceCache.end())
    if (!cacheSource(LineInfo))
      return {};

  auto LineBuffer = LineCache.find(LineInfo.FileName);
  if (LineBuffer == LineCache.end())
    return {};

  if (LineInfo.Line > LineBuffer->second.size()) {
    reportWarning(
        formatv("debug info line number {0} exceeds the number of lines in {1}",
                LineInfo.Line, LineInfo.FileName),
        ObjectFilename);
    return {};
  }

  // Vector begins at 0, line numbers are non-zero
  return LineBuffer->second[LineInfo.Line - 1];
}

void SourcePrinter::printSources(formatted_raw_ostream &OS,
                                 const DILineInfo &LineInfo,
                                 StringRef ObjectFilename, StringRef Delimiter,
                                 LiveVariablePrinter &LVP) {
  if (LineInfo.FileName == DILineInfo::BadString || LineInfo.Line == 0 ||
      (OldLineInfo.Line == LineInfo.Line &&
       OldLineInfo.FileName == LineInfo.FileName))
    return;

  StringRef Line = getLine(LineInfo, ObjectFilename);
  if (!Line.empty()) {
    OS << Delimiter << Line;
    LVP.printBetweenInsts(OS, true);
  }
}

SourcePrinter::SourcePrinter(const object::ObjectFile *Obj,
                             StringRef DefaultArch)
    : Obj(Obj) {
  symbolize::LLVMSymbolizer::Options SymbolizerOpts;
  SymbolizerOpts.PrintFunctions =
      DILineInfoSpecifier::FunctionNameKind::LinkageName;
  SymbolizerOpts.Demangle = Demangle;
  SymbolizerOpts.DefaultArch = std::string(DefaultArch);
  Symbolizer.reset(new symbolize::LLVMSymbolizer(SymbolizerOpts));
}

// TODO Light/dark shades? 256-color terminals?
const raw_ostream::Colors LineColors[] = {
    raw_ostream::RED,  raw_ostream::GREEN,   raw_ostream::YELLOW,
    raw_ostream::BLUE, raw_ostream::MAGENTA, raw_ostream::CYAN,
};

raw_ostream::Colors ControlFlowPrinter::PickColor() {
  if (!OutputMode.color_enabled())
    return raw_ostream::RESET;
  auto Ret = LineColors[NextColorIdx];
  NextColorIdx =
      (NextColorIdx + 1) % (sizeof(LineColors) / sizeof(LineColors[0]));
  return Ret;
}

void ControlFlowPrinter::addEdge(uint64_t From, uint64_t To) {
  auto It = Targets.find(To);
  if (It == Targets.end())
    It = Targets.insert(std::make_pair(To, ControlFlowTarget(To, PickColor())))
             .first;
  It->second.addSource(From);
}

void ControlFlowPrinter::finalise() {
  if (!OutputMode.enabled()) {
    setControlFlowColumnWidth(0);
    return;
  }

  SmallVector<ControlFlowTarget *> SortedTargets;
  for (auto &[Addr, Info] : Targets) {
    SortedTargets.push_back(&Info);
  }
  std::sort(SortedTargets.begin(), SortedTargets.end(),
            [](ControlFlowTarget *LHS, ControlFlowTarget *RHS) {
              return LHS->Length() < RHS->Length();
            });

  // FIXME This is O(n^3) in the worst case, can we do better?
  for (auto &T : SortedTargets) {
    int Column = 0;
    for (;; ++Column)
      if (!std::any_of(SortedTargets.begin(), SortedTargets.end(),
                       [T, Column](ControlFlowTarget *T2) {
                         return T != T2 && T2->Column == Column &&
                                T->Overlaps(*T2);
                       }))
        break;
    T->Column = Column;
    MaxColumn = std::max(MaxColumn, Column);
  }

  setControlFlowColumnWidth(MaxColumn * 2 + 4);
}

const char *ControlFlowPrinter::getLineChar(LineChar C) const {
  bool IsASCII = !OutputMode.unicode_enabled();
  switch (C) {
  case LineChar::Horiz:
    return IsASCII ? "-" : u8"\u2500";
  case LineChar::Vert:
    return IsASCII ? "|" : u8"\u2502";
  case LineChar::TopCorner:
    return IsASCII ? "/" : u8"\u256d";
  case LineChar::BottomCorner:
    return IsASCII ? "\\" : u8"\u2570";
  case LineChar::Tee:
    return IsASCII ? "+" : u8"\u251c";
  case LineChar::Arrow:
    return ">";
  }
  llvm_unreachable("Unhandled LineChar enum");
}

#define C(id) getLineChar(LineChar::id)

void ControlFlowPrinter::printInst(formatted_raw_ostream &OS,
                                   uint64_t Addr) const {
  if (!OutputMode.enabled())
    return;

  SmallVector<const ControlFlowTarget *, 8> Columns;
  Columns.resize(MaxColumn + 1);
  const ControlFlowTarget *Horizontal = nullptr;

  IndentToColumn(STI, OS, DisassemblyColumn::ControlFlow);

  // TODO: What happens if an instruction has both incoming and outgoing edges?

  for (auto &[_, Info] : Targets) {
    if (Info.ActiveAt(Addr)) {
      assert(Columns[Info.Column] == nullptr);
      Columns[Info.Column] = &Info;
    }
  }

  auto Color = [&](raw_ostream &OS,
                   raw_ostream::Colors Color) -> raw_ostream & {
    if (OutputMode.color_enabled()) {
      OS << Color;
    }
    return OS;
  };

  OS.PadToColumn(getIndentLevel());
  for (int ColIdx = MaxColumn; ColIdx >= 0; --ColIdx) {
    if (Horizontal) {
      if (Columns[ColIdx]) {
        // Outgoing horizontal lines are drawn "under" vertical line, because
        // that minimises the gap in the "lower" line.
        Color(OS, Horizontal->Color) << C(Horiz);
        Color(OS, Columns[ColIdx]->Color) << C(Vert);
      } else
        Color(OS, Horizontal->Color) << C(Horiz) << C(Horiz);
    } else if (!Columns[ColIdx])
      OS << "  ";
    else if (Columns[ColIdx]->HorizontalAt(Addr)) {
      Horizontal = Columns[ColIdx];
      if (Columns[ColIdx]->StartsAt(Addr))
        Color(OS, Horizontal->Color) << " " << C(TopCorner);
      else if (Columns[ColIdx]->EndsAt(Addr))
        Color(OS, Horizontal->Color) << " " << C(BottomCorner);
      else
        Color(OS, Horizontal->Color) << " " << C(Tee);
    } else if (Columns[ColIdx]->ActiveAt(Addr))
      Color(OS, Columns[ColIdx]->Color) << " " << C(Vert);
    else
      OS << "  ";
  }

  if (Horizontal) {
    if (Horizontal->TargetAt(Addr))
      Color(OS, Horizontal->Color) << C(Horiz) << C(Arrow);
    else
      Color(OS, Horizontal->Color) << C(Horiz) << C(Horiz);
  } else {
    OS << "  ";
  }

  Color(OS, raw_ostream::RESET);
}

// TODO boolean params -> enum?
void ControlFlowPrinter::printOther(formatted_raw_ostream &OS, uint64_t Addr,
                                    bool BeforeInst, bool AfterInst) const {
  if (!OutputMode.enabled())
    return;

  assert(!(BeforeInst && AfterInst));

  SmallVector<const ControlFlowTarget *, 8> Columns;
  Columns.resize(MaxColumn + 1);

  auto Color = [&](raw_ostream &OS,
                   raw_ostream::Colors Color) -> raw_ostream & {
    if (OutputMode.color_enabled()) {
      OS << Color;
    }
    return OS;
  };

  IndentToColumn(STI, OS, DisassemblyColumn::ControlFlow);

  for (auto &[_, Info] : Targets) {
    if (Info.ActiveAt(Addr, BeforeInst, AfterInst)) {
      assert(Columns[Info.Column] == nullptr);
      Columns[Info.Column] = &Info;
    }
  }

  OS.PadToColumn(getIndentLevel());
  for (int ColIdx = MaxColumn; ColIdx >= 0; --ColIdx) {
    if (!Columns[ColIdx])
      OS << "  ";
    else
      Color(OS, Columns[ColIdx]->Color) << " " << C(Vert);
  }

  Color(OS, raw_ostream::RESET) << "  ";
}

#undef C

} // namespace objdump
} // namespace llvm

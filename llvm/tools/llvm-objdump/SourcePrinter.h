//===-- SourcePrinter.h -  source interleaving utilities --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_OBJDUMP_SOURCEPRINTER_H
#define LLVM_TOOLS_LLVM_OBJDUMP_SOURCEPRINTER_H

#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/FormattedStream.h"
#include <unordered_map>
#include <vector>

namespace llvm {
namespace objdump {

class ControlFlowPrinter;

/// Stores a single expression representing the location of a source-level
/// variable, along with the PC range for which that expression is valid.
struct LiveVariable {
  DWARFLocationExpression LocExpr;
  const char *VarName;
  DWARFUnit *Unit;
  const DWARFDie FuncDie;

  LiveVariable(const DWARFLocationExpression &LocExpr, const char *VarName,
               DWARFUnit *Unit, const DWARFDie FuncDie)
      : LocExpr(LocExpr), VarName(VarName), Unit(Unit), FuncDie(FuncDie) {}

  bool liveAtAddress(object::SectionedAddress Addr);

  void print(raw_ostream &OS, const MCRegisterInfo &MRI) const;
};

/// Helper class for printing source variable locations alongside disassembly.
class LiveVariablePrinter {
  // Information we want to track about one column in which we are printing a
  // variable live range.
  struct Column {
    unsigned VarIdx = NullVarIdx;
    bool LiveIn = false;
    bool LiveOut = false;
    bool MustDrawLabel = false;

    bool isActive() const { return VarIdx != NullVarIdx; }

    static constexpr unsigned NullVarIdx = std::numeric_limits<unsigned>::max();
  };

  // All live variables we know about in the object/image file.
  std::vector<LiveVariable> LiveVariables;

  // The columns we are currently drawing.
  IndexedMap<Column> ActiveCols;

  const MCRegisterInfo &MRI;
  const MCSubtargetInfo &STI;

  void addVariable(DWARFDie FuncDie, DWARFDie VarDie);

  void addFunction(DWARFDie D);

  // Get the column number (in characters) at which the first live variable
  // line should be printed.
  unsigned getIndentLevel() const;

  // Indent to the first live-range column to the right of the currently
  // printed line, and return the index of that column.
  // TODO: formatted_raw_ostream uses "column" to mean a number of characters
  // since the last \n, and we use it to mean the number of slots in which we
  // put live variable lines. Pick a less overloaded word.
  unsigned moveToFirstVarColumn(formatted_raw_ostream &OS);

  unsigned findFreeColumn();

public:
  LiveVariablePrinter(const MCRegisterInfo &MRI, const MCSubtargetInfo &STI)
      : ActiveCols(Column()), MRI(MRI), STI(STI) {}

  void dump() const;

  void addCompileUnit(DWARFDie D);

  /// Update to match the state of the instruction between ThisAddr and
  /// NextAddr. In the common case, any live range active at ThisAddr is
  /// live-in to the instruction, and any live range active at NextAddr is
  /// live-out of the instruction. If IncludeDefinedVars is false, then live
  /// ranges starting at NextAddr will be ignored.
  void update(object::SectionedAddress ThisAddr,
              object::SectionedAddress NextAddr, bool IncludeDefinedVars);

  enum class LineChar {
    RangeStart,
    RangeMid,
    RangeEnd,
    LabelVert,
    LabelCornerNew,
    LabelCornerActive,
    LabelHoriz,
  };
  const char *getLineChar(LineChar C) const;

  /// Print live ranges to the right of an existing line. This assumes the
  /// line is not an instruction, so doesn't start or end any live ranges, so
  /// we only need to print active ranges or empty columns. If AfterInst is
  /// true, this is being printed after the last instruction fed to update(),
  /// otherwise this is being printed before it.
  void printAfterOtherLine(formatted_raw_ostream &OS, bool AfterInst);

  /// Print any live variable range info needed to the right of a
  /// non-instruction line of disassembly. This is where we print the variable
  /// names and expressions, with thin line-drawing characters connecting them
  /// to the live range which starts at the next instruction. If MustPrint is
  /// true, we have to print at least one line (with the continuation of any
  /// already-active live ranges) because something has already been printed
  /// earlier on this line.
  void printBetweenInsts(formatted_raw_ostream &OS, bool MustPrint,
                         uint64_t Addr = 0, ControlFlowPrinter *CFP = nullptr);

  /// Print the live variable ranges to the right of a disassembled instruction.
  void printAfterInst(formatted_raw_ostream &OS);
};

class SourcePrinter {
protected:
  DILineInfo OldLineInfo;
  const object::ObjectFile *Obj = nullptr;
  std::unique_ptr<symbolize::LLVMSymbolizer> Symbolizer;
  // File name to file contents of source.
  std::unordered_map<std::string, std::unique_ptr<MemoryBuffer>> SourceCache;
  // Mark the line endings of the cached source.
  std::unordered_map<std::string, std::vector<StringRef>> LineCache;
  // Keep track of missing sources.
  StringSet<> MissingSources;
  // Only emit 'invalid debug info' warning once.
  bool WarnedInvalidDebugInfo = false;

private:
  bool cacheSource(const DILineInfo &LineInfoFile);

  void printLines(formatted_raw_ostream &OS, const DILineInfo &LineInfo,
                  StringRef Delimiter, LiveVariablePrinter &LVP);

  void printSources(formatted_raw_ostream &OS, const DILineInfo &LineInfo,
                    StringRef ObjectFilename, StringRef Delimiter,
                    LiveVariablePrinter &LVP);

  // Returns line source code corresponding to `LineInfo`.
  // Returns empty string if source code cannot be found.
  StringRef getLine(const DILineInfo &LineInfo, StringRef ObjectFilename);

public:
  SourcePrinter() = default;
  SourcePrinter(const object::ObjectFile *Obj, StringRef DefaultArch);
  virtual ~SourcePrinter() = default;
  virtual void printSourceLine(formatted_raw_ostream &OS,
                               object::SectionedAddress Address,
                               StringRef ObjectFilename,
                               LiveVariablePrinter &LVP,
                               StringRef Delimiter = "; ");
};

struct VisualizeJumpsMode {
  enum Chars_t { Off, ASCII, Unicode };
  enum Colors_t { BlackAndWhite, ThreeBit, Auto };

  Chars_t Chars;
  Colors_t Colors;

  VisualizeJumpsMode() : Chars(Off), Colors(BlackAndWhite) {}
  VisualizeJumpsMode(Chars_t Chars, Colors_t Colors)
      : Chars(Chars), Colors(Colors) {}

  static VisualizeJumpsMode GetDefault() {
    return VisualizeJumpsMode(Unicode, Auto);
  }

  bool enabled() const { return Chars != Off; }
  bool color_enabled() const { return enabled() && Colors != BlackAndWhite; }
  bool unicode_enabled() const { return Chars == Unicode; }

  void ResolveAutoColor(raw_ostream &OS) {
    if (Colors == Auto)
      Colors = OS.has_colors() ? ThreeBit : BlackAndWhite;
  }
};

class ControlFlowPrinter {
  struct ControlFlowTarget {
    uint64_t Target;
    SmallVector<uint64_t, 4> Sources;
    int Column;
    raw_ostream::Colors Color;

    ControlFlowTarget(uint64_t Target, raw_ostream::Colors Color)
        : Target(Target), Column(~0U), Color(Color), High(Target), Low(Target) {
    }
    ControlFlowTarget(const ControlFlowTarget &) = delete;
    ControlFlowTarget(ControlFlowTarget &&) = default;

    void addSource(uint64_t Source) {
      Sources.push_back(Source);
      Low = std::min(Low, Source);
      High = std::max(High, Source);
    }

    uint64_t Length() const { return High - Low; }

    bool Overlaps(ControlFlowTarget &Other) const {
      return !(Other.Low > High || Other.High < Low);
    }

    bool ActiveAt(uint64_t Addr, bool BeforeInst = false,
                  bool AfterInst = false) const {
      if (BeforeInst)
        return Addr > Low && Addr <= High;
      else if (AfterInst)
        return Addr >= Low && Addr < High;
      else
        return Addr >= Low && Addr <= High;
    }

    bool StartsAt(uint64_t Addr) const { return Addr == Low; }
    bool EndsAt(uint64_t Addr) const { return Addr == High; }
    bool TargetAt(uint64_t Addr) const { return Addr == Target; }

    bool HorizontalAt(uint64_t Addr) const {
      return Addr == Target ||
             std::any_of(Sources.begin(), Sources.end(),
                         [Addr](uint64_t Src) { return Src == Addr; });
    }

  private:
    uint64_t High, Low;
  };

  VisualizeJumpsMode OutputMode;
  DenseMap<uint64_t, ControlFlowTarget> Targets;
  int MaxColumn;
  const MCSubtargetInfo &STI;

  int NextColorIdx;
  raw_ostream::Colors PickColor();

  int getIndentLevel() const { return 10; }

  enum class LineChar {
    Horiz,
    Vert,
    TopCorner,
    BottomCorner,
    Tee,
    Arrow,
  };
  const char *getLineChar(LineChar C) const;

public:
  ControlFlowPrinter(VisualizeJumpsMode OutputMode, const MCSubtargetInfo &STI)
      : OutputMode(OutputMode), MaxColumn(0), STI(STI), NextColorIdx(0) {}

  // Add a control-flow edge from the instruction at address From to the
  // instruction at address To.
  void addEdge(uint64_t From, uint64_t To);

  void finalise();

  void printInst(formatted_raw_ostream &OS, uint64_t Addr) const;
  void printOther(formatted_raw_ostream &OS, uint64_t Addr,
                  bool BeforeInst = false, bool AfterInst = false) const;
};

} // namespace objdump
} // namespace llvm

#endif

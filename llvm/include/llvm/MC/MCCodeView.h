//===- MCCodeView.h - Machine Code CodeView support -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Holds state from .cv_file and .cv_loc directives for later emission.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCCODEVIEW_H
#define LLVM_MC_MCCODEVIEW_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCFragment.h"
#include <map>
#include <vector>

namespace llvm {
class MCContext;
class MCObjectStreamer;
class MCStreamer;

/// \brief Instances of this class represent the information from a
/// .cv_loc directive.
class MCCVLoc {
  uint32_t FunctionId;
  uint32_t FileNum;
  uint32_t Line;
  uint16_t Column;
  uint16_t PrologueEnd : 1;
  uint16_t IsStmt : 1;

private: // MCContext manages these
  friend class MCContext;
  MCCVLoc(unsigned functionid, unsigned fileNum, unsigned line, unsigned column,
          bool prologueend, bool isstmt)
      : FunctionId(functionid), FileNum(fileNum), Line(line), Column(column),
        PrologueEnd(prologueend), IsStmt(isstmt) {}

  // Allow the default copy constructor and assignment operator to be used
  // for an MCCVLoc object.

public:
  unsigned getFunctionId() const { return FunctionId; }

  /// \brief Get the FileNum of this MCCVLoc.
  unsigned getFileNum() const { return FileNum; }

  /// \brief Get the Line of this MCCVLoc.
  unsigned getLine() const { return Line; }

  /// \brief Get the Column of this MCCVLoc.
  unsigned getColumn() const { return Column; }

  bool isPrologueEnd() const { return PrologueEnd; }
  bool isStmt() const { return IsStmt; }

  void setFunctionId(unsigned FID) { FunctionId = FID; }

  /// \brief Set the FileNum of this MCCVLoc.
  void setFileNum(unsigned fileNum) { FileNum = fileNum; }

  /// \brief Set the Line of this MCCVLoc.
  void setLine(unsigned line) { Line = line; }

  /// \brief Set the Column of this MCCVLoc.
  void setColumn(unsigned column) {
    assert(column <= UINT16_MAX);
    Column = column;
  }

  void setPrologueEnd(bool PE) { PrologueEnd = PE; }
  void setIsStmt(bool IS) { IsStmt = IS; }
};

/// \brief Instances of this class represent the line information for
/// the CodeView line table entries.  Which is created after a machine
/// instruction is assembled and uses an address from a temporary label
/// created at the current address in the current section and the info from
/// the last .cv_loc directive seen as stored in the context.
class MCCVLineEntry : public MCCVLoc {
  MCSymbol *Label;

private:
  // Allow the default copy constructor and assignment operator to be used
  // for an MCCVLineEntry object.

public:
  // Constructor to create an MCCVLineEntry given a symbol and the dwarf loc.
  MCCVLineEntry(MCSymbol *label, const MCCVLoc loc)
      : MCCVLoc(loc), Label(label) {}

  MCSymbol *getLabel() const { return Label; }

  // This is called when an instruction is assembled into the specified
  // section and if there is information from the last .cv_loc directive that
  // has yet to have a line entry made for it is made.
  static void Make(MCObjectStreamer *MCOS);
};

/// Holds state from .cv_file and .cv_loc directives for later emission.
class CodeViewContext {
public:
  CodeViewContext();
  ~CodeViewContext();

  bool isValidFileNumber(unsigned FileNumber) const;
  bool addFile(unsigned FileNumber, StringRef Filename);
  ArrayRef<StringRef> getFilenames() { return Filenames; }

  /// \brief Add a line entry.
  void addLineEntry(const MCCVLineEntry &LineEntry) {
    size_t Offset = MCCVLines.size();
    auto I =
        MCCVLineStartStop.insert({LineEntry.getFunctionId(), {Offset, Offset}});
    if (!I.second)
      I.first->second.second = Offset;
    MCCVLines.push_back(LineEntry);
  }

  std::vector<MCCVLineEntry> getFunctionLineEntries(unsigned FuncId) {
    std::vector<MCCVLineEntry> FilteredLines;

    auto I = MCCVLineStartStop.find(FuncId);
    if (I != MCCVLineStartStop.end())
      for (size_t Idx = I->second.first, End = I->second.second + 1; Idx != End;
           ++Idx)
        if (MCCVLines[Idx].getFunctionId() == FuncId)
          FilteredLines.push_back(MCCVLines[Idx]);
    return FilteredLines;
  }

  /// Emits a line table substream.
  void emitLineTableForFunction(MCObjectStreamer &OS, unsigned FuncId,
                                const MCSymbol *FuncBegin,
                                const MCSymbol *FuncEnd);

  void emitInlineLineTableForFunction(MCObjectStreamer &OS,
                                      unsigned PrimaryFunctionId,
                                      unsigned SourceFileId,
                                      unsigned SourceLineNum,
                                      ArrayRef<unsigned> SecondaryFunctionIds);

  /// Emits the string table substream.
  void emitStringTable(MCObjectStreamer &OS);

  /// Emits the file checksum substream.
  void emitFileChecksums(MCObjectStreamer &OS);

private:
  /// Map from string to string table offset.
  StringMap<unsigned> StringTable;

  /// The fragment that ultimately holds our strings.
  MCDataFragment *StrTabFragment = nullptr;
  bool InsertedStrTabFragment = false;

  MCDataFragment *getStringTableFragment();

  /// Add something to the string table.
  StringRef addToStringTable(StringRef S);

  /// Get a string table offset.
  unsigned getStringTableOffset(StringRef S);

  /// An array of absolute paths. Eventually this may include the file checksum.
  SmallVector<StringRef, 4> Filenames;

  /// The offset of the first and last .cv_loc directive for a given function
  /// id.
  std::map<unsigned, std::pair<size_t, size_t>> MCCVLineStartStop;

  /// A collection of MCCVLineEntry for each section.
  std::vector<MCCVLineEntry> MCCVLines;
};

} // end namespace llvm
#endif

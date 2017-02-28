//===- DIContext.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines DIContext, an abstract data structure that holds
// debug information data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DICONTEXT_H
#define LLVM_DEBUGINFO_DICONTEXT_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Object/ObjectFile.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

namespace llvm {

class raw_ostream;

/// DILineInfo - a format-neutral container for source line information.
struct DILineInfo {
  std::string FileName;
  std::string FunctionName;
  uint32_t Line = 0;
  uint32_t Column = 0;
  uint32_t StartLine = 0;

  // DWARF-specific.
  uint32_t Discriminator = 0;

  DILineInfo() : FileName("<invalid>"), FunctionName("<invalid>") {}

  bool operator==(const DILineInfo &RHS) const {
    return Line == RHS.Line && Column == RHS.Column &&
           FileName == RHS.FileName && FunctionName == RHS.FunctionName &&
           StartLine == RHS.StartLine && Discriminator == RHS.Discriminator;
  }
  bool operator!=(const DILineInfo &RHS) const {
    return !(*this == RHS);
  }
  bool operator<(const DILineInfo &RHS) const {
    return std::tie(FileName, FunctionName, Line, Column, StartLine,
                    Discriminator) <
           std::tie(RHS.FileName, RHS.FunctionName, RHS.Line, RHS.Column,
                    RHS.StartLine, RHS.Discriminator);
  }
};

typedef SmallVector<std::pair<uint64_t, DILineInfo>, 16> DILineInfoTable;

/// DIInliningInfo - a format-neutral container for inlined code description.
class DIInliningInfo {
  SmallVector<DILineInfo, 4> Frames;

public:
  DIInliningInfo() = default;

  DILineInfo getFrame(unsigned Index) const {
    assert(Index < Frames.size());
    return Frames[Index];
  }

  DILineInfo *getMutableFrame(unsigned Index) {
    assert(Index < Frames.size());
    return &Frames[Index];
  }

  uint32_t getNumberOfFrames() const {
    return Frames.size();
  }

  void addFrame(const DILineInfo &Frame) {
    Frames.push_back(Frame);
  }
};

/// DIGlobal - container for description of a global variable.
struct DIGlobal {
  std::string Name;
  uint64_t Start = 0;
  uint64_t Size = 0;

  DIGlobal() : Name("<invalid>") {}
};

/// A DINameKind is passed to name search methods to specify a
/// preference regarding the type of name resolution the caller wants.
enum class DINameKind { None, ShortName, LinkageName };

/// DILineInfoSpecifier - controls which fields of DILineInfo container
/// should be filled with data.
struct DILineInfoSpecifier {
  enum class FileLineInfoKind { None, Default, AbsoluteFilePath };
  typedef DINameKind FunctionNameKind;

  FileLineInfoKind FLIKind;
  FunctionNameKind FNKind;

  DILineInfoSpecifier(FileLineInfoKind FLIKind = FileLineInfoKind::Default,
                      FunctionNameKind FNKind = FunctionNameKind::None)
      : FLIKind(FLIKind), FNKind(FNKind) {}
};

/// Selects which debug sections get dumped.
enum DIDumpType {
  DIDT_Null,
  DIDT_All,
  DIDT_Abbrev,
  DIDT_AbbrevDwo,
  DIDT_Aranges,
  DIDT_Frames,
  DIDT_Info,
  DIDT_InfoDwo,
  DIDT_Types,
  DIDT_TypesDwo,
  DIDT_Line,
  DIDT_LineDwo,
  DIDT_Loc,
  DIDT_LocDwo,
  DIDT_Macro,
  DIDT_Ranges,
  DIDT_Pubnames,
  DIDT_Pubtypes,
  DIDT_GnuPubnames,
  DIDT_GnuPubtypes,
  DIDT_Str,
  DIDT_StrDwo,
  DIDT_StrOffsetsDwo,
  DIDT_AppleNames,
  DIDT_AppleTypes,
  DIDT_AppleNamespaces,
  DIDT_AppleObjC,
  DIDT_CUIndex,
  DIDT_GdbIndex,
  DIDT_TUIndex,
};

class DIContext {
public:
  enum DIContextKind {
    CK_DWARF,
    CK_PDB
  };

  DIContext(DIContextKind K) : Kind(K) {}
  virtual ~DIContext() = default;

  DIContextKind getKind() const { return Kind; }

  virtual void dump(raw_ostream &OS, DIDumpType DumpType = DIDT_All,
                    bool DumpEH = false, bool SummarizeTypes = false) = 0;

  virtual DILineInfo getLineInfoForAddress(uint64_t Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) = 0;
  virtual DILineInfoTable getLineInfoForAddressRange(uint64_t Address,
      uint64_t Size, DILineInfoSpecifier Specifier = DILineInfoSpecifier()) = 0;
  virtual DIInliningInfo getInliningInfoForAddress(uint64_t Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) = 0;

private:
  const DIContextKind Kind;
};

/// An inferface for inquiring the load address of a loaded object file
/// to be used by the DIContext implementations when applying relocations
/// on the fly.
class LoadedObjectInfo {
protected:
  LoadedObjectInfo() = default;
  LoadedObjectInfo(const LoadedObjectInfo &) = default;

public:
  virtual ~LoadedObjectInfo() = default;

  /// Obtain the Load Address of a section by SectionRef.
  ///
  /// Calculate the address of the given section.
  /// The section need not be present in the local address space. The addresses
  /// need to be consistent with the addresses used to query the DIContext and
  /// the output of this function should be deterministic, i.e. repeated calls with
  /// the same Sec should give the same address.
  virtual uint64_t getSectionLoadAddress(const object::SectionRef &Sec) const = 0;

  /// If conveniently available, return the content of the given Section.
  ///
  /// When the section is available in the local address space, in relocated (loaded)
  /// form, e.g. because it was relocated by a JIT for execution, this function
  /// should provide the contents of said section in `Data`. If the loaded section
  /// is not available, or the cost of retrieving it would be prohibitive, this
  /// function should return false. In that case, relocations will be read from the
  /// local (unrelocated) object file and applied on the fly. Note that this method
  /// is used purely for optimzation purposes in the common case of JITting in the
  /// local address space, so returning false should always be correct.
  virtual bool getLoadedSectionContents(const object::SectionRef &Sec,
                                        StringRef &Data) const {
    return false;
  }

  /// Obtain a copy of this LoadedObjectInfo.
  ///
  /// The caller is responsible for deallocation once the copy is no longer required.
  virtual std::unique_ptr<LoadedObjectInfo> clone() const = 0;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DICONTEXT_H

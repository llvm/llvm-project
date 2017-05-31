//===-- WindowsResource.h ---------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This file declares the .res file class.  .res files are intermediate
// products of the typical resource-compilation process on Windows.  This
// process is as follows:
//
// .rc file(s) ---(rc.exe)---> .res file(s) ---(cvtres.exe)---> COFF file
//
// .rc files are human-readable scripts that list all resources a program uses.
//
// They are compiled into .res files, which are a list of the resources in
// binary form.
//
// Finally the data stored in the .res is compiled into a COFF file, where it
// is organized in a directory tree structure for optimized access by the
// program during runtime.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/ms648007(v=vs.85).aspx
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_INCLUDE_LLVM_OBJECT_RESFILE_H
#define LLVM_INCLUDE_LLVM_OBJECT_RESFILE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"

#include <map>

namespace llvm {
namespace object {

class WindowsResource;

class ResourceEntryRef {
public:
  Error moveNext(bool &End);
  bool checkTypeString() const { return IsStringType; }
  ArrayRef<UTF16> getTypeString() const { return Type; }
  uint16_t getTypeID() const { return TypeID; }
  bool checkNameString() const { return IsStringName; }
  ArrayRef<UTF16> getNameString() const { return Name; }
  uint16_t getNameID() const { return NameID; }
  uint16_t getLanguage() const { return Suffix->Language; }

private:
  friend class WindowsResource;

  ResourceEntryRef(BinaryStreamRef Ref, const WindowsResource *Owner,
                   Error &Err);

  Error loadNext();

  struct HeaderSuffix {
    support::ulittle32_t DataVersion;
    support::ulittle16_t MemoryFlags;
    support::ulittle16_t Language;
    support::ulittle32_t Version;
    support::ulittle32_t Characteristics;
  };

  BinaryStreamReader Reader;
  bool IsStringType;
  ArrayRef<UTF16> Type;
  uint16_t TypeID;
  bool IsStringName;
  ArrayRef<UTF16> Name;
  uint16_t NameID;
  const HeaderSuffix *Suffix = nullptr;
  ArrayRef<uint8_t> Data;
  const WindowsResource *OwningRes = nullptr;
};

class WindowsResource : public Binary {
public:
  Expected<ResourceEntryRef> getHeadEntry();

  static bool classof(const Binary *V) { return V->isWinRes(); }

  static Expected<std::unique_ptr<WindowsResource>>
  createWindowsResource(MemoryBufferRef Source);

private:
  friend class ResourceEntryRef;

  WindowsResource(MemoryBufferRef Source);

  BinaryByteStream BBS;
};

class WindowsResourceParser {
public:
  WindowsResourceParser();

  Error parse(WindowsResource *WR);

  void printTree() const;

private:
  class TreeNode {
  public:
    TreeNode() = default;
    explicit TreeNode(ArrayRef<UTF16> Ref);
    void addEntry(const ResourceEntryRef &Entry);
    void print(ScopedPrinter &Writer, StringRef Name) const;

  private:
    TreeNode &addTypeNode(const ResourceEntryRef &Entry);
    TreeNode &addNameNode(const ResourceEntryRef &Entry);
    TreeNode &addLanguageNode(const ResourceEntryRef &Entry);
    TreeNode &addChild(uint32_t ID);
    TreeNode &addChild(ArrayRef<UTF16> NameRef);
    std::vector<UTF16> Name;
    std::map<uint32_t, std::unique_ptr<TreeNode>> IDChildren;
    std::map<std::string, std::unique_ptr<TreeNode>> StringChildren;
  };

  TreeNode Root;
};

} // namespace object
} // namespace llvm

#endif

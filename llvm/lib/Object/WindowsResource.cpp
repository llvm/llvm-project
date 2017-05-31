//===-- WindowsResource.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the .res file class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/WindowsResource.h"
#include "llvm/Support/COFF.h"
#include <sstream>
#include <system_error>

namespace llvm {
namespace object {

#define RETURN_IF_ERROR(X)                                                     \
  if (auto EC = X)                                                             \
    return EC;

const uint32_t MIN_HEADER_SIZE = 7 * sizeof(uint32_t) + 2 * sizeof(uint16_t);

static const size_t ResourceMagicSize = 16;

static const size_t NullEntrySize = 16;

WindowsResource::WindowsResource(MemoryBufferRef Source)
    : Binary(Binary::ID_WinRes, Source) {
  size_t LeadingSize = ResourceMagicSize + NullEntrySize;
  BBS = BinaryByteStream(Data.getBuffer().drop_front(LeadingSize),
                         support::little);
}

Expected<std::unique_ptr<WindowsResource>>
WindowsResource::createWindowsResource(MemoryBufferRef Source) {
  if (Source.getBufferSize() < ResourceMagicSize + NullEntrySize)
    return make_error<GenericBinaryError>(
        "File too small to be a resource file",
        object_error::invalid_file_type);
  std::unique_ptr<WindowsResource> Ret(new WindowsResource(Source));
  return std::move(Ret);
}

Expected<ResourceEntryRef> WindowsResource::getHeadEntry() {
  Error Err = Error::success();
  auto Ref = ResourceEntryRef(BinaryStreamRef(BBS), this, Err);
  if (Err)
    return std::move(Err);
  return Ref;
}

ResourceEntryRef::ResourceEntryRef(BinaryStreamRef Ref,
                                   const WindowsResource *Owner, Error &Err)
    : Reader(Ref), OwningRes(Owner) {
  if (loadNext())
    Err = make_error<GenericBinaryError>("Could not read first entry.",
                                         object_error::unexpected_eof);
}

Error ResourceEntryRef::moveNext(bool &End) {
  // Reached end of all the entries.
  if (Reader.bytesRemaining() == 0) {
    End = true;
    return Error::success();
  }
  RETURN_IF_ERROR(loadNext());

  return Error::success();
}

static Error readStringOrId(BinaryStreamReader &Reader, uint16_t &ID,
                            ArrayRef<UTF16> &Str, bool &IsString) {
  uint16_t IDFlag;
  RETURN_IF_ERROR(Reader.readInteger(IDFlag));
  IsString = IDFlag != 0xffff;

  if (IsString) {
    Reader.setOffset(
        Reader.getOffset() -
        sizeof(uint16_t)); // Re-read the bytes which we used to check the flag.
    RETURN_IF_ERROR(Reader.readWideString(Str));
  } else
    RETURN_IF_ERROR(Reader.readInteger(ID));

  return Error::success();
}

Error ResourceEntryRef::loadNext() {
  uint32_t DataSize;
  RETURN_IF_ERROR(Reader.readInteger(DataSize));
  uint32_t HeaderSize;
  RETURN_IF_ERROR(Reader.readInteger(HeaderSize));

  if (HeaderSize < MIN_HEADER_SIZE)
    return make_error<GenericBinaryError>("Header size is too small.",
                                          object_error::parse_failed);

  RETURN_IF_ERROR(readStringOrId(Reader, TypeID, Type, IsStringType));

  RETURN_IF_ERROR(readStringOrId(Reader, NameID, Name, IsStringName));

  RETURN_IF_ERROR(Reader.padToAlignment(sizeof(uint32_t)));

  RETURN_IF_ERROR(Reader.readObject(Suffix));

  RETURN_IF_ERROR(Reader.readArray(Data, DataSize));

  RETURN_IF_ERROR(Reader.padToAlignment(sizeof(uint32_t)));

  return Error::success();
}

WindowsResourceParser::WindowsResourceParser() {}

Error WindowsResourceParser::parse(WindowsResource *WR) {
  auto EntryOrErr = WR->getHeadEntry();
  if (!EntryOrErr)
    return EntryOrErr.takeError();

  ResourceEntryRef Entry = EntryOrErr.get();
  bool End = false;

  while (!End) {

    Root.addEntry(Entry);

    RETURN_IF_ERROR(Entry.moveNext(End));
  }

  return Error::success();
}

void WindowsResourceParser::printTree() const {
  ScopedPrinter Writer(outs());
  Root.print(Writer, "Resource Tree");
}

void WindowsResourceParser::TreeNode::addEntry(const ResourceEntryRef &Entry) {
  TreeNode &TypeNode = addTypeNode(Entry);
  TreeNode &NameNode = TypeNode.addNameNode(Entry);
  NameNode.addLanguageNode(Entry);
}

WindowsResourceParser::TreeNode::TreeNode(ArrayRef<UTF16> NameRef)
    : Name(NameRef) {}

WindowsResourceParser::TreeNode &
WindowsResourceParser::TreeNode::addTypeNode(const ResourceEntryRef &Entry) {
  if (Entry.checkTypeString())
    return addChild(Entry.getTypeString());
  else
    return addChild(Entry.getTypeID());
}

WindowsResourceParser::TreeNode &
WindowsResourceParser::TreeNode::addNameNode(const ResourceEntryRef &Entry) {
  if (Entry.checkNameString())
    return addChild(Entry.getNameString());
  else
    return addChild(Entry.getNameID());
}

WindowsResourceParser::TreeNode &
WindowsResourceParser::TreeNode::addLanguageNode(
    const ResourceEntryRef &Entry) {
  return addChild(Entry.getLanguage());
}

WindowsResourceParser::TreeNode &
WindowsResourceParser::TreeNode::addChild(uint32_t ID) {
  auto Child = IDChildren.find(ID);
  if (Child == IDChildren.end()) {
    auto NewChild = llvm::make_unique<WindowsResourceParser::TreeNode>(ID);
    WindowsResourceParser::TreeNode &Node = *NewChild;
    IDChildren.emplace(ID, std::move(NewChild));
    return Node;
  } else
    return *(Child->second);
}

WindowsResourceParser::TreeNode &
WindowsResourceParser::TreeNode::addChild(ArrayRef<UTF16> NameRef) {
  std::string NameString;
  ArrayRef<UTF16> CorrectedName;
  std::vector<UTF16> EndianCorrectedName;
  if (llvm::sys::IsBigEndianHost) {
    EndianCorrectedName.resize(NameRef.size() + 1);
    std::copy(NameRef.begin(), NameRef.end(), EndianCorrectedName.begin() + 1);
    EndianCorrectedName[0] = UNI_UTF16_BYTE_ORDER_MARK_SWAPPED;
    CorrectedName = makeArrayRef(EndianCorrectedName);
  } else
    CorrectedName = NameRef;
  llvm::convertUTF16ToUTF8String(CorrectedName, NameString);

  auto Child = StringChildren.find(NameString);
  if (Child == StringChildren.end()) {
    auto NewChild = llvm::make_unique<WindowsResourceParser::TreeNode>(NameRef);
    WindowsResourceParser::TreeNode &Node = *NewChild;
    StringChildren.emplace(NameString, std::move(NewChild));
    return Node;
  } else
    return *(Child->second);
}

void WindowsResourceParser::TreeNode::print(ScopedPrinter &Writer,
                                            StringRef Name) const {
  ListScope NodeScope(Writer, Name);
  for (auto const &Child : StringChildren) {
    Child.second->print(Writer, Child.first);
  }
  for (auto const &Child : IDChildren) {
    Child.second->print(Writer, to_string(Child.first));
  }
}

} // namespace object
} // namespace llvm

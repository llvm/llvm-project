//===-- ResourceSerializator.h ----------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This defines a visitor serializing resources to a .res stream.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMRC_RESOURCESERIALIZATOR_H
#define LLVM_TOOLS_LLVMRC_RESOURCESERIALIZATOR_H

#include "ResourceScriptStmt.h"
#include "ResourceVisitor.h"

#include "llvm/Support/Endian.h"

namespace llvm {

class MemoryBuffer;

namespace rc {

struct SearchParams {
  std::vector<std::string> Include;   // Additional folders to search for files.
  std::vector<std::string> NoInclude; // Folders to exclude from file search.
  StringRef InputFilePath;            // The full path of the input file.
};

class ResourceFileWriter : public Visitor {
public:
  ResourceFileWriter(const SearchParams &Params,
                     std::unique_ptr<raw_fd_ostream> Stream)
      : Params(Params), FS(std::move(Stream)), IconCursorID(1) {
    assert(FS && "Output stream needs to be provided to the serializator");
  }

  Error visitNullResource(const RCResource *) override;
  Error visitAcceleratorsResource(const RCResource *) override;
  Error visitCursorResource(const RCResource *) override;
  Error visitDialogResource(const RCResource *) override;
  Error visitHTMLResource(const RCResource *) override;
  Error visitIconResource(const RCResource *) override;
  Error visitMenuResource(const RCResource *) override;
  Error visitVersionInfoResource(const RCResource *) override;
  Error visitStringTableResource(const RCResource *) override;
  Error visitUserDefinedResource(const RCResource *) override;

  Error visitCaptionStmt(const CaptionStmt *) override;
  Error visitCharacteristicsStmt(const CharacteristicsStmt *) override;
  Error visitFontStmt(const FontStmt *) override;
  Error visitLanguageStmt(const LanguageResource *) override;
  Error visitStyleStmt(const StyleStmt *) override;
  Error visitVersionStmt(const VersionStmt *) override;

  // Stringtables are output at the end of .res file. We need a separate
  // function to do it.
  Error dumpAllStringTables();

  bool AppendNull; // Append '\0' to each existing STRINGTABLE element?

  struct ObjectInfo {
    uint16_t LanguageInfo;
    uint32_t Characteristics;
    uint32_t VersionInfo;

    Optional<uint32_t> Style;
    StringRef Caption;
    struct FontInfo {
      uint32_t Size;
      StringRef Typeface;
      uint32_t Weight;
      bool IsItalic;
      uint32_t Charset;
    };
    Optional<FontInfo> Font;

    ObjectInfo() : LanguageInfo(0), Characteristics(0), VersionInfo(0) {}
  } ObjectData;

  struct StringTableInfo {
    // Each STRINGTABLE bundle depends on ID of the bundle and language
    // description.
    using BundleKey = std::pair<uint16_t, uint16_t>;
    // Each bundle is in fact an array of 16 strings.
    struct Bundle {
      std::array<Optional<StringRef>, 16> Data;
      ObjectInfo DeclTimeInfo;
      Bundle(const ObjectInfo &Info) : DeclTimeInfo(Info) {}
    };
    std::map<BundleKey, Bundle> BundleData;
    // Bundles are listed in the order of their first occurrence.
    std::vector<BundleKey> BundleList;
  } StringTableData;

private:
  Error handleError(Error Err, const RCResource *Res);

  Error
  writeResource(const RCResource *Res,
                Error (ResourceFileWriter::*BodyWriter)(const RCResource *));

  // NullResource
  Error writeNullBody(const RCResource *);

  // AcceleratorsResource
  Error writeSingleAccelerator(const AcceleratorsResource::Accelerator &,
                               bool IsLastItem);
  Error writeAcceleratorsBody(const RCResource *);

  // CursorResource and IconResource
  Error visitIconOrCursorResource(const RCResource *);
  Error visitIconOrCursorGroup(const RCResource *);
  Error visitSingleIconOrCursor(const RCResource *);
  Error writeSingleIconOrCursorBody(const RCResource *);
  Error writeIconOrCursorGroupBody(const RCResource *);

  // DialogResource
  Error writeSingleDialogControl(const Control &, bool IsExtended);
  Error writeDialogBody(const RCResource *);

  // HTMLResource
  Error writeHTMLBody(const RCResource *);

  // MenuResource
  Error writeMenuDefinition(const std::unique_ptr<MenuDefinition> &,
                            uint16_t Flags);
  Error writeMenuDefinitionList(const MenuDefinitionList &List);
  Error writeMenuBody(const RCResource *);

  // StringTableResource
  Error visitStringTableBundle(const RCResource *);
  Error writeStringTableBundleBody(const RCResource *);
  Error insertStringIntoBundle(StringTableInfo::Bundle &Bundle,
                               uint16_t StringID, StringRef String);

  // User defined resource
  Error writeUserDefinedBody(const RCResource *);

  // VersionInfoResource
  Error writeVersionInfoBody(const RCResource *);
  Error writeVersionInfoBlock(const VersionInfoBlock &);
  Error writeVersionInfoValue(const VersionInfoValue &);

  const SearchParams &Params;

  // Output stream handling.
  std::unique_ptr<raw_fd_ostream> FS;

  uint64_t tell() const { return FS->tell(); }

  uint64_t writeObject(const ArrayRef<uint8_t> Data);

  template <typename T> uint64_t writeInt(const T &Value) {
    support::detail::packed_endian_specific_integral<T, support::little,
                                                     support::unaligned>
        Object(Value);
    return writeObject(Object);
  }

  template <typename T> uint64_t writeObject(const T &Value) {
    return writeObject(ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(&Value), sizeof(T)));
  }

  template <typename T> void writeObjectAt(const T &Value, uint64_t Position) {
    FS->pwrite((const char *)&Value, sizeof(T), Position);
  }

  Error writeCString(StringRef Str, bool WriteTerminator = true);

  Error writeIdentifier(const IntOrString &Ident);
  Error writeIntOrString(const IntOrString &Data);

  void writeRCInt(RCInt);

  Error appendFile(StringRef Filename);

  void padStream(uint64_t Length);

  Expected<std::unique_ptr<MemoryBuffer>> loadFile(StringRef File) const;

  // Icon and cursor IDs are allocated starting from 1 and increasing for
  // each icon/cursor dumped. This maintains the current ID to be allocated.
  uint16_t IconCursorID;
};

} // namespace rc
} // namespace llvm

#endif

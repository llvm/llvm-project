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
namespace rc {

class ResourceFileWriter : public Visitor {
public:
  ResourceFileWriter(std::unique_ptr<raw_fd_ostream> Stream)
      : FS(std::move(Stream)) {
    assert(FS && "Output stream needs to be provided to the serializator");
  }

  Error visitNullResource(const RCResource *) override;
  Error visitAcceleratorsResource(const RCResource *) override;
  Error visitHTMLResource(const RCResource *) override;

  Error visitCharacteristicsStmt(const CharacteristicsStmt *) override;
  Error visitLanguageStmt(const LanguageResource *) override;
  Error visitVersionStmt(const VersionStmt *) override;

  struct ObjectInfo {
    uint16_t LanguageInfo;
    uint32_t Characteristics;
    uint32_t VersionInfo;

    ObjectInfo() : LanguageInfo(0), Characteristics(0), VersionInfo(0) {}
  } ObjectData;

private:
  Error handleError(Error &&Err, const RCResource *Res);

  Error
  writeResource(const RCResource *Res,
                Error (ResourceFileWriter::*BodyWriter)(const RCResource *));

  // NullResource
  Error writeNullBody(const RCResource *);

  // AcceleratorsResource
  Error writeSingleAccelerator(const AcceleratorsResource::Accelerator &,
                               bool IsLastItem);
  Error writeAcceleratorsBody(const RCResource *);

  // HTMLResource
  Error writeHTMLBody(const RCResource *);

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

  Error appendFile(StringRef Filename);

  void padStream(uint64_t Length);
};

} // namespace rc
} // namespace llvm

#endif

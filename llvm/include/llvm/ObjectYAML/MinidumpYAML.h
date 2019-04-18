//===- MinidumpYAML.h - Minidump YAMLIO implementation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_MINIDUMPYAML_H
#define LLVM_OBJECTYAML_MINIDUMPYAML_H

#include "llvm/BinaryFormat/Minidump.h"
#include "llvm/Object/Minidump.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace MinidumpYAML {

/// The base class for all minidump streams. The "Type" of the stream
/// corresponds to the Stream Type field in the minidump file. The "Kind" field
/// specifies how are we going to treat it. For highly specialized streams (e.g.
/// SystemInfo), there is a 1:1 mapping between Types and Kinds, but in general
/// one stream Kind can be used to represent multiple stream Types (e.g. any
/// unrecognised stream Type will be handled via RawContentStream). The mapping
/// from Types to Kinds is fixed and given by the static getKind function.
struct Stream {
  enum class StreamKind {
    ModuleList,
    RawContent,
    SystemInfo,
    TextContent,
  };

  Stream(StreamKind Kind, minidump::StreamType Type) : Kind(Kind), Type(Type) {}
  virtual ~Stream(); // anchor

  const StreamKind Kind;
  const minidump::StreamType Type;

  /// Get the stream Kind used for representing streams of a given Type.
  static StreamKind getKind(minidump::StreamType Type);

  /// Create an empty stream of the given Type.
  static std::unique_ptr<Stream> create(minidump::StreamType Type);

  /// Create a stream from the given stream directory entry.
  static Expected<std::unique_ptr<Stream>>
  create(const minidump::Directory &StreamDesc,
         const object::MinidumpFile &File);
};

/// A stream representing the list of modules loaded in the process. On disk, it
/// is represented as a sequence of minidump::Module structures. These contain
/// pointers to other data structures, like the module's name and CodeView
/// record. In memory, we represent these as the ParsedModule struct, which
/// groups minidump::Module with all of its dependant structures in a single
/// entity.
struct ModuleListStream : public Stream {
  struct ParsedModule {
    minidump::Module Module;
    std::string Name;
    yaml::BinaryRef CvRecord;
    yaml::BinaryRef MiscRecord;
  };
  std::vector<ParsedModule> Modules;

  ModuleListStream(std::vector<ParsedModule> Modules = {})
      : Stream(StreamKind::ModuleList, minidump::StreamType::ModuleList),
        Modules(std::move(Modules)) {}

  static bool classof(const Stream *S) {
    return S->Kind == StreamKind::ModuleList;
  }
};

/// A minidump stream represented as a sequence of hex bytes. This is used as a
/// fallback when no other stream kind is suitable.
struct RawContentStream : public Stream {
  yaml::BinaryRef Content;
  yaml::Hex32 Size;

  RawContentStream(minidump::StreamType Type, ArrayRef<uint8_t> Content = {})
      : Stream(StreamKind::RawContent, Type), Content(Content),
        Size(Content.size()) {}

  static bool classof(const Stream *S) {
    return S->Kind == StreamKind::RawContent;
  }
};

/// SystemInfo minidump stream.
struct SystemInfoStream : public Stream {
  minidump::SystemInfo Info;
  std::string CSDVersion;

  explicit SystemInfoStream(const minidump::SystemInfo &Info,
                            std::string CSDVersion)
      : Stream(StreamKind::SystemInfo, minidump::StreamType::SystemInfo),
        Info(Info), CSDVersion(std::move(CSDVersion)) {}

  SystemInfoStream()
      : Stream(StreamKind::SystemInfo, minidump::StreamType::SystemInfo) {
    memset(&Info, 0, sizeof(Info));
  }

  static bool classof(const Stream *S) {
    return S->Kind == StreamKind::SystemInfo;
  }
};

/// A StringRef, which is printed using YAML block notation.
LLVM_YAML_STRONG_TYPEDEF(StringRef, BlockStringRef)

/// A minidump stream containing textual data (typically, the contents of a
/// /proc/<pid> file on linux).
struct TextContentStream : public Stream {
  BlockStringRef Text;

  TextContentStream(minidump::StreamType Type, StringRef Text = {})
      : Stream(StreamKind::TextContent, Type), Text(Text) {}

  static bool classof(const Stream *S) {
    return S->Kind == StreamKind::TextContent;
  }
};

/// The top level structure representing a minidump object, consisting of a
/// minidump header, and zero or more streams. To construct an Object from a
/// minidump file, use the static create function. To serialize to/from yaml,
/// use the appropriate streaming operator on a yaml stream.
struct Object {
  Object() = default;
  Object(const Object &) = delete;
  Object &operator=(const Object &) = delete;
  Object(Object &&) = default;
  Object &operator=(Object &&) = default;

  Object(const minidump::Header &Header,
         std::vector<std::unique_ptr<Stream>> Streams)
      : Header(Header), Streams(std::move(Streams)) {}

  /// The minidump header.
  minidump::Header Header;

  /// The list of streams in this minidump object.
  std::vector<std::unique_ptr<Stream>> Streams;

  static Expected<Object> create(const object::MinidumpFile &File);
};

/// Serialize the minidump file represented by Obj to OS in binary form.
void writeAsBinary(Object &Obj, raw_ostream &OS);

/// Serialize the yaml string as a minidump file to OS in binary form.
Error writeAsBinary(StringRef Yaml, raw_ostream &OS);

} // namespace MinidumpYAML

namespace yaml {
template <> struct BlockScalarTraits<MinidumpYAML::BlockStringRef> {
  static void output(const MinidumpYAML::BlockStringRef &Text, void *,
                     raw_ostream &OS) {
    OS << Text;
  }

  static StringRef input(StringRef Scalar, void *,
                         MinidumpYAML::BlockStringRef &Text) {
    Text = Scalar;
    return "";
  }
};

template <> struct MappingTraits<std::unique_ptr<MinidumpYAML::Stream>> {
  static void mapping(IO &IO, std::unique_ptr<MinidumpYAML::Stream> &S);
  static StringRef validate(IO &IO, std::unique_ptr<MinidumpYAML::Stream> &S);
};

} // namespace yaml

} // namespace llvm

LLVM_YAML_DECLARE_ENUM_TRAITS(llvm::minidump::ProcessorArchitecture)
LLVM_YAML_DECLARE_ENUM_TRAITS(llvm::minidump::OSPlatform)
LLVM_YAML_DECLARE_ENUM_TRAITS(llvm::minidump::StreamType)

LLVM_YAML_DECLARE_MAPPING_TRAITS(llvm::minidump::CPUInfo::ArmInfo)
LLVM_YAML_DECLARE_MAPPING_TRAITS(llvm::minidump::CPUInfo::OtherInfo)
LLVM_YAML_DECLARE_MAPPING_TRAITS(llvm::minidump::CPUInfo::X86Info)
LLVM_YAML_DECLARE_MAPPING_TRAITS(llvm::minidump::VSFixedFileInfo)
LLVM_YAML_DECLARE_MAPPING_TRAITS(
    llvm::MinidumpYAML::ModuleListStream::ParsedModule)

LLVM_YAML_IS_SEQUENCE_VECTOR(std::unique_ptr<llvm::MinidumpYAML::Stream>)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::MinidumpYAML::ModuleListStream::ParsedModule)

LLVM_YAML_DECLARE_MAPPING_TRAITS(llvm::MinidumpYAML::Object)

#endif // LLVM_OBJECTYAML_MINIDUMPYAML_H

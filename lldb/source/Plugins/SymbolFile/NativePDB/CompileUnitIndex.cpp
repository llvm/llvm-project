//===-- CompileUnitIndex.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CompileUnitIndex.h"

#include "PdbIndex.h"
#include "PdbUtil.h"

#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptor.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Native/NamedStreamMap.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/Support/Path.h"

#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::npdb;
using namespace llvm::codeview;
using namespace llvm::pdb;

static bool IsMainFile(llvm::StringRef main, llvm::StringRef other) {
  if (main == other)
    return true;

  // If the files refer to the local file system, we can just ask the file
  // system if they're equivalent.  But if the source isn't present on disk
  // then we still want to try.
  if (llvm::sys::fs::equivalent(main, other))
    return true;

  llvm::SmallString<64> normalized(other);
  llvm::sys::path::native(normalized);
  return main.equals_insensitive(normalized);
}

static llvm::Error ParseCompile3(const CVSymbol &sym, CompilandIndexItem &cci) {
  cci.m_compile_opts.emplace();
  if (auto err = SymbolDeserializer::deserializeAs<Compile3Sym>(
          sym, *cci.m_compile_opts)) {
    cci.m_compile_opts.reset();
    return err;
  }
  return llvm::Error::success();
}

static llvm::Error ParseObjname(const CVSymbol &sym, CompilandIndexItem &cci) {
  cci.m_obj_name.emplace();
  if (auto err =
          SymbolDeserializer::deserializeAs<ObjNameSym>(sym, *cci.m_obj_name)) {
    cci.m_obj_name.reset();
    return err;
  }
  return llvm::Error::success();
}

static llvm::Error ParseBuildInfo(PdbIndex &index, const CVSymbol &sym,
                                  CompilandIndexItem &cci) {
  BuildInfoSym bis(SymbolRecordKind::BuildInfoSym);
  if (auto err = SymbolDeserializer::deserializeAs<BuildInfoSym>(sym, bis))
    return err;

  // S_BUILDINFO just points to an LF_BUILDINFO in the IPI stream.  Let's do
  // a little extra work to pull out the LF_BUILDINFO.
  LazyRandomTypeCollection &types = index.ipi().typeCollection();
  std::optional<CVType> cvt = types.tryGetType(bis.BuildId);

  if (!cvt || cvt->kind() != LF_BUILDINFO)
    return llvm::Error::success();

  BuildInfoRecord bir;
  if (auto err = TypeDeserializer::deserializeAs<BuildInfoRecord>(*cvt, bir))
    return err;
  cci.m_build_info.assign(bir.ArgIndices.begin(), bir.ArgIndices.end());
  return llvm::Error::success();
}

static void ParseExtendedInfo(PdbIndex &index, CompilandIndexItem &item) {
  const CVSymbolArray &syms = item.m_debug_stream.getSymbolArray();

  // This is a private function, it shouldn't be called if the information
  // has already been parsed.
  lldbassert(!item.m_obj_name);
  lldbassert(!item.m_compile_opts);
  lldbassert(item.m_build_info.empty());

  Log *log = GetLog(LLDBLog::Symbols);
  // We're looking for 3 things.  S_COMPILE3, S_OBJNAME, and S_BUILDINFO.
  int found = 0;
  for (const CVSymbol &sym : syms) {
    switch (sym.kind()) {
    case S_COMPILE3:
      if (auto err = ParseCompile3(sym, item))
        LLDB_LOG_ERROR(log, std::move(err),
                       "Failed to parse S_COMPILE3 record: {0}");
      break;
    case S_OBJNAME:
      if (auto err = ParseObjname(sym, item))
        LLDB_LOG_ERROR(log, std::move(err),
                       "Failed to parse S_OBJNAME record: {0}");
      break;
    case S_BUILDINFO:
      if (auto err = ParseBuildInfo(index, sym, item))
        LLDB_LOG_ERROR(log, std::move(err),
                       "Failed to parse S_BUILDINFO record: {0}");
      break;
    default:
      continue;
    }
    if (++found >= 3)
      break;
  }
}

static void ParseInlineeLineTableForCompileUnit(CompilandIndexItem &item) {
  Log *log = GetLog(LLDBLog::Symbols);
  for (const auto &ss : item.m_debug_stream.getSubsectionsArray()) {
    if (ss.kind() != DebugSubsectionKind::InlineeLines)
      continue;

    DebugInlineeLinesSubsectionRef inlinee_lines;
    llvm::BinaryStreamReader reader(ss.getRecordData());
    if (llvm::Error error = inlinee_lines.initialize(reader)) {
      LLDB_LOG_ERROR(log, std::move(error),
                     "Failed to initialize inlinee lines subsection: {0}");
      continue;
    }

    for (const InlineeSourceLine &Line : inlinee_lines) {
      item.m_inline_map[Line.Header->Inlinee] = Line;
    }
  }
}

CompilandIndexItem::CompilandIndexItem(
    PdbCompilandId id, llvm::pdb::ModuleDebugStreamRef debug_stream,
    llvm::pdb::DbiModuleDescriptor descriptor)
    : m_id(id), m_debug_stream(std::move(debug_stream)),
      m_module_descriptor(std::move(descriptor)) {}

CompilandIndexItem &CompileUnitIndex::GetOrCreateCompiland(uint16_t modi) {
  auto result = m_comp_units.try_emplace(modi, nullptr);
  if (!result.second)
    return *result.first->second;

  // Find the module list and load its debug information stream and cache it
  // since we need to use it for almost all interesting operations.
  const DbiModuleList &modules = m_index.dbi().modules();
  llvm::pdb::DbiModuleDescriptor descriptor = modules.getModuleDescriptor(modi);
  uint16_t stream = descriptor.getModuleStreamIndex();
  std::unique_ptr<llvm::msf::MappedBlockStream> stream_data =
      m_index.pdb().createIndexedStream(stream);


  std::unique_ptr<CompilandIndexItem>& cci = result.first->second;

  if (!stream_data) {
    llvm::pdb::ModuleDebugStreamRef debug_stream(descriptor, nullptr);
    cci = std::make_unique<CompilandIndexItem>(PdbCompilandId{ modi }, debug_stream, std::move(descriptor));
    return *cci;
  }

  llvm::pdb::ModuleDebugStreamRef debug_stream(descriptor,
                                               std::move(stream_data));

  if (llvm::Error err = debug_stream.reload()) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), std::move(err),
                   "Failed to reload debug stream for module {1}: {0}", modi);
    llvm::pdb::ModuleDebugStreamRef empty_stream(descriptor, nullptr);
    cci = std::make_unique<CompilandIndexItem>(
        PdbCompilandId{modi}, empty_stream, std::move(descriptor));
    return *cci;
  }

  cci = std::make_unique<CompilandIndexItem>(
      PdbCompilandId{modi}, std::move(debug_stream), std::move(descriptor));
  ParseExtendedInfo(m_index, *cci);
  ParseInlineeLineTableForCompileUnit(*cci);

  auto strings = m_index.pdb().getStringTable();
  if (strings) {
    cci->m_strings.initialize(cci->m_debug_stream.getSubsectionsArray());
    cci->m_strings.setStrings(strings->getStringTable());
  } else {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), strings.takeError(),
                   "Failed to get PDB string table: {0}");
  }

  // We want the main source file to always comes first.  Note that we can't
  // just push_back the main file onto the front because `GetMainSourceFile`
  // computes it in such a way that it doesn't own the resulting memory.  So we
  // have to iterate the module file list comparing each one to the main file
  // name until we find it, and we can cache that one since the memory is backed
  // by a contiguous chunk inside the mapped PDB.
  llvm::SmallString<64> main_file;
  if (auto main_file_or_err = GetMainSourceFile(*cci)) {
    main_file = std::move(*main_file_or_err);
  } else {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Symbols), main_file_or_err.takeError(),
                   "Failed to determine main source file for module {1}: {0}",
                   modi);
  }
  llvm::sys::path::native(main_file);

  uint32_t file_count = modules.getSourceFileCount(modi);
  cci->m_file_list.reserve(file_count);
  bool found_main_file = false;
  for (llvm::StringRef file : modules.source_files(modi)) {
    if (!found_main_file && IsMainFile(main_file, file)) {
      cci->m_file_list.insert(cci->m_file_list.begin(), file);
      found_main_file = true;
      continue;
    }
    cci->m_file_list.push_back(file);
  }

  return *cci;
}

const CompilandIndexItem *CompileUnitIndex::GetCompiland(uint16_t modi) const {
  auto iter = m_comp_units.find(modi);
  if (iter == m_comp_units.end())
    return nullptr;
  return iter->second.get();
}

CompilandIndexItem *CompileUnitIndex::GetCompiland(uint16_t modi) {
  auto iter = m_comp_units.find(modi);
  if (iter == m_comp_units.end())
    return nullptr;
  return iter->second.get();
}

llvm::Expected<llvm::SmallString<64>>
CompileUnitIndex::GetMainSourceFile(const CompilandIndexItem &item) const {
  // LF_BUILDINFO contains a list of arg indices which point to LF_STRING_ID
  // records in the IPI stream.  The order of the arg indices is as follows:
  // [0] - working directory where compiler was invoked.
  // [1] - absolute path to compiler binary
  // [2] - source file name
  // [3] - path to compiler generated PDB (the /Zi PDB, although this entry gets
  //       added even when using /Z7)
  // [4] - full command line invocation.
  //
  // We need to form the path [0]\[2] to generate the full path to the main
  // file.source
  if (item.m_build_info.size() < 3)
    return llvm::SmallString<64>("");

  LazyRandomTypeCollection &types = m_index.ipi().typeCollection();

  StringIdRecord working_dir;
  StringIdRecord file_name;
  CVType dir_cvt = types.getType(item.m_build_info[0]);
  CVType file_cvt = types.getType(item.m_build_info[2]);
  if (auto err =
          TypeDeserializer::deserializeAs<StringIdRecord>(dir_cvt, working_dir))
    return std::move(err);
  if (auto err =
          TypeDeserializer::deserializeAs<StringIdRecord>(file_cvt, file_name))
    return std::move(err);

  llvm::sys::path::Style style = working_dir.String.starts_with("/")
                                     ? llvm::sys::path::Style::posix
                                     : llvm::sys::path::Style::windows;
  if (llvm::sys::path::is_absolute(file_name.String, style))
    return file_name.String;

  llvm::SmallString<64> absolute_path = working_dir.String;
  llvm::sys::path::append(absolute_path, file_name.String);
  return absolute_path;
}

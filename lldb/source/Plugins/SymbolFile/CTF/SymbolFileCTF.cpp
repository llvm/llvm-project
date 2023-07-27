//===-- SymbolFileCTF.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolFileCTF.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamBuffer.h"
#include "lldb/Host/Config.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/TypeMap.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timer.h"
#include "llvm/Support/MemoryBuffer.h"

#include "Plugins/ExpressionParser/Clang/ClangASTMetadata.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#include <memory>
#include <optional>

#if LLVM_ENABLE_ZLIB
#include <zlib.h>
#endif

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(SymbolFileCTF)

char SymbolFileCTF::ID;

SymbolFileCTF::SymbolFileCTF(lldb::ObjectFileSP objfile_sp)
    : SymbolFileCommon(std::move(objfile_sp)) {}

void SymbolFileCTF::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void SymbolFileCTF::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef SymbolFileCTF::GetPluginDescriptionStatic() {
  return "Compact C Type Format Symbol Reader";
}

SymbolFile *SymbolFileCTF::CreateInstance(ObjectFileSP objfile_sp) {
  return new SymbolFileCTF(std::move(objfile_sp));
}

bool SymbolFileCTF::ParseHeader() {
  if (m_header)
    return true;

  Log *log = GetLog(LLDBLog::Symbols);

  ModuleSP module_sp(m_objfile_sp->GetModule());
  const SectionList *section_list = module_sp->GetSectionList();
  if (!section_list)
    return false;

  SectionSP section_sp(
      section_list->FindSectionByType(lldb::eSectionTypeCTF, true));
  if (!section_sp)
    return false;

  m_objfile_sp->ReadSectionData(section_sp.get(), m_data);

  if (m_data.GetByteSize() == 0)
    return false;

  StreamString module_desc;
  GetObjectFile()->GetModule()->GetDescription(module_desc.AsRawOstream(),
                                               lldb::eDescriptionLevelBrief);
  LLDB_LOG(log, "Parsing Compact C Type format for {0}", module_desc.GetData());

  lldb::offset_t offset = 0;

  // Parse CTF header.
  constexpr size_t ctf_header_size = sizeof(ctf_header_t);
  if (!m_data.ValidOffsetForDataOfSize(offset, ctf_header_size)) {
    LLDB_LOG(log, "CTF parsing failed: insufficient data for CTF header");
    return false;
  }

  m_header.emplace();

  ctf_header_t &ctf_header = *m_header;
  ctf_header.preamble.magic = m_data.GetU16(&offset);
  ctf_header.preamble.version = m_data.GetU8(&offset);
  ctf_header.preamble.flags = m_data.GetU8(&offset);
  ctf_header.parlabel = m_data.GetU32(&offset);
  ctf_header.parname = m_data.GetU32(&offset);
  ctf_header.lbloff = m_data.GetU32(&offset);
  ctf_header.objtoff = m_data.GetU32(&offset);
  ctf_header.funcoff = m_data.GetU32(&offset);
  ctf_header.typeoff = m_data.GetU32(&offset);
  ctf_header.stroff = m_data.GetU32(&offset);
  ctf_header.strlen = m_data.GetU32(&offset);

  // Validate the preamble.
  if (ctf_header.preamble.magic != g_ctf_magic) {
    LLDB_LOG(log, "CTF parsing failed: invalid magic: {0:x}",
             ctf_header.preamble.magic);
    return false;
  }

  if (ctf_header.preamble.version != g_ctf_version) {
    LLDB_LOG(log, "CTF parsing failed: unsupported version: {0}",
             ctf_header.preamble.version);
    return false;
  }

  LLDB_LOG(log, "Parsed valid CTF preamble: version {0}, flags {1:x}",
           ctf_header.preamble.version, ctf_header.preamble.flags);

  m_body_offset = offset;

  if (ctf_header.preamble.flags & eFlagCompress) {
    // The body has been compressed with zlib deflate. Header offsets point into
    // the decompressed data.
#if LLVM_ENABLE_ZLIB
    const std::size_t decompressed_size = ctf_header.stroff + ctf_header.strlen;
    DataBufferSP decompressed_data =
        std::make_shared<DataBufferHeap>(decompressed_size, 0x0);

    z_stream zstr;
    memset(&zstr, 0, sizeof(zstr));
    zstr.next_in = (Bytef *)const_cast<uint8_t *>(m_data.GetDataStart() +
                                                  sizeof(ctf_header_t));
    zstr.avail_in = m_data.BytesLeft(offset);
    zstr.next_out =
        (Bytef *)const_cast<uint8_t *>(decompressed_data->GetBytes());
    zstr.avail_out = decompressed_size;

    int rc = inflateInit(&zstr);
    if (rc != Z_OK) {
      LLDB_LOG(log, "CTF parsing failed: inflate initialization error: {0}",
               zError(rc));
      return false;
    }

    rc = inflate(&zstr, Z_FINISH);
    if (rc != Z_STREAM_END) {
      LLDB_LOG(log, "CTF parsing failed: inflate error: {0}", zError(rc));
      return false;
    }

    rc = inflateEnd(&zstr);
    if (rc != Z_OK) {
      LLDB_LOG(log, "CTF parsing failed: inflate end error: {0}", zError(rc));
      return false;
    }

    if (zstr.total_out != decompressed_size) {
      LLDB_LOG(log,
               "CTF parsing failed: decompressed size ({0}) doesn't match "
               "expected size ([1})",
               zstr.total_out, decompressed_size);
      return false;
    }

    m_data = DataExtractor(decompressed_data, m_data.GetByteOrder(),
                           m_data.GetAddressByteSize());
    m_body_offset = 0;
#else
    LLDB_LOG(
        log,
        "CTF parsing failed: data is compressed but no zlib inflate support");
    return false;
#endif
  }

  // Validate the header.
  if (!m_data.ValidOffset(m_body_offset + ctf_header.lbloff)) {
    LLDB_LOG(log,
             "CTF parsing failed: invalid label section offset in header: {0}",
             ctf_header.lbloff);
    return false;
  }

  if (!m_data.ValidOffset(m_body_offset + ctf_header.objtoff)) {
    LLDB_LOG(log,
             "CTF parsing failed: invalid object section offset in header: {0}",
             ctf_header.objtoff);
    return false;
  }

  if (!m_data.ValidOffset(m_body_offset + ctf_header.funcoff)) {
    LLDB_LOG(
        log,
        "CTF parsing failed: invalid function section offset in header: {0}",
        ctf_header.funcoff);
    return false;
  }

  if (!m_data.ValidOffset(m_body_offset + ctf_header.typeoff)) {
    LLDB_LOG(log,
             "CTF parsing failed: invalid type section offset in header: {0}",
             ctf_header.typeoff);
    return false;
  }

  if (!m_data.ValidOffset(m_body_offset + ctf_header.stroff)) {
    LLDB_LOG(log,
             "CTF parsing failed: invalid string section offset in header: {0}",
             ctf_header.stroff);
    return false;
  }

  const lldb::offset_t str_end_offset =
      m_body_offset + ctf_header.stroff + ctf_header.strlen;
  if (!m_data.ValidOffset(str_end_offset - 1)) {
    LLDB_LOG(log,
             "CTF parsing failed: invalid string section length in header: {0}",
             ctf_header.strlen);
    return false;
  }

  if (m_body_offset + ctf_header.stroff + ctf_header.parlabel >
      str_end_offset) {
    LLDB_LOG(log,
             "CTF parsing failed: invalid parent label offset: {0} exceeds end "
             "of string section ({1})",
             ctf_header.parlabel, str_end_offset);
    return false;
  }

  if (m_body_offset + ctf_header.stroff + ctf_header.parname > str_end_offset) {
    LLDB_LOG(log,
             "CTF parsing failed: invalid parent name offset: {0} exceeds end "
             "of string section ({1})",
             ctf_header.parname, str_end_offset);
    return false;
  }

  LLDB_LOG(log,
           "Parsed valid CTF header: lbloff  = {0}, objtoff = {1}, funcoff = "
           "{2}, typeoff = {3}, stroff = {4}, strlen = {5}",
           ctf_header.lbloff, ctf_header.objtoff, ctf_header.funcoff,
           ctf_header.typeoff, ctf_header.stroff, ctf_header.strlen);

  return true;
}

void SymbolFileCTF::InitializeObject() {
  Log *log = GetLog(LLDBLog::Symbols);

  auto type_system_or_err = GetTypeSystemForLanguage(lldb::eLanguageTypeC);
  if (auto err = type_system_or_err.takeError()) {
    LLDB_LOG_ERROR(log, std::move(err), "Unable to get type system: {0}");
    return;
  }

  auto ts = *type_system_or_err;
  m_ast = llvm::dyn_cast_or_null<TypeSystemClang>(ts.get());
  LazyBool optimized = eLazyBoolNo;
  m_comp_unit_sp = std::make_shared<CompileUnit>(
      m_objfile_sp->GetModule(), nullptr, "", 0, eLanguageTypeC, optimized);

  ParseTypes(*m_comp_unit_sp);
}

llvm::StringRef SymbolFileCTF::ReadString(lldb::offset_t str_offset) const {
  lldb::offset_t offset = m_body_offset + m_header->stroff + str_offset;
  if (!m_data.ValidOffset(offset))
    return "(invalid)";
  const char *str = m_data.GetCStr(&offset);
  if (str && !*str)
    return "(anon)";
  return llvm::StringRef(str);
}

/// Return the integer display representation encoded in the given data.
static uint32_t GetEncoding(uint32_t data) {
  // Mask bits 24–31.
  return ((data)&0xff000000) >> 24;
}

/// Return the integral width in bits encoded in the given data.
static uint32_t GetBits(uint32_t data) {
  // Mask bits 0-15.
  return (data)&0x0000ffff;
}

/// Return the type kind encoded in the given data.
uint32_t GetKind(uint32_t data) {
  // Mask bits 26–31.
  return ((data)&0xf800) >> 11;
}

/// Return the variable length encoded in the given data.
uint32_t GetVLen(uint32_t data) {
  // Mask bits 0–24.
  return (data)&0x3ff;
}

static uint32_t GetBytes(uint32_t bits) { return bits / sizeof(unsigned); }

static clang::TagTypeKind TranslateRecordKind(SymbolFileCTF::TypeKind type) {
  switch (type) {
  case SymbolFileCTF::TypeKind::eStruct:
    return clang::TTK_Struct;
  case SymbolFileCTF::TypeKind::eUnion:
    return clang::TTK_Union;
  default:
    lldbassert(false && "Invalid record kind!");
    return clang::TTK_Struct;
  }
}

llvm::Expected<TypeSP> SymbolFileCTF::ParseInteger(lldb::offset_t &offset,
                                                   lldb::user_id_t uid,
                                                   llvm::StringRef name) {
  const uint32_t vdata = m_data.GetU32(&offset);
  const uint32_t bits = GetBits(vdata);
  const uint32_t encoding = GetEncoding(vdata);

  lldb::BasicType basic_type = TypeSystemClang::GetBasicTypeEnumeration(name);
  if (basic_type == eBasicTypeInvalid)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("unsupported integer type: no corresponding basic clang "
                      "type for '{0}'",
                      name),
        llvm::inconvertibleErrorCode());

  CompilerType compiler_type = m_ast->GetBasicType(basic_type);

  if (basic_type != eBasicTypeVoid) {
    // Make sure the type we got is an integer type.
    bool compiler_type_is_signed = false;
    if (!compiler_type.IsIntegerType(compiler_type_is_signed))
      return llvm::make_error<llvm::StringError>(
          llvm::formatv(
              "Found compiler type for '{0}' but it's not an integer type: {1}",
              name, compiler_type.GetDisplayTypeName().GetStringRef()),
          llvm::inconvertibleErrorCode());

    // Make sure the signing matches between the CTF and the compiler type.
    const bool type_is_signed = (encoding & IntEncoding::eSigned);
    if (compiler_type_is_signed != type_is_signed)
      return llvm::make_error<llvm::StringError>(
          llvm::formatv("Found integer compiler type for {0} but compiler type "
                        "is {1} and {0} is {2}",
                        name, compiler_type_is_signed ? "signed" : "unsigned",
                        type_is_signed ? "signed" : "unsigned"),
          llvm::inconvertibleErrorCode());
  }

  Declaration decl;
  return MakeType(uid, ConstString(name), GetBytes(bits), nullptr,
                  LLDB_INVALID_UID, lldb_private::Type::eEncodingIsUID, decl,
                  compiler_type, lldb_private::Type::ResolveState::Full);
}

llvm::Expected<lldb::TypeSP>
SymbolFileCTF::ParseModifierType(lldb::offset_t &offset, lldb::user_id_t uid,
                                 uint32_t kind, uint32_t type) {
  TypeSP ref_type = GetTypeForUID(type);
  if (!ref_type)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Could not find modified type: {0}", type),
        llvm::inconvertibleErrorCode());

  CompilerType compiler_type;

  switch (kind) {
  case TypeKind::ePointer:
    compiler_type = ref_type->GetFullCompilerType().GetPointerType();
    break;
  case TypeKind::eConst:
    compiler_type = ref_type->GetFullCompilerType().AddConstModifier();
    break;
  case TypeKind::eVolatile:
    compiler_type = ref_type->GetFullCompilerType().AddVolatileModifier();
    break;
  case TypeKind::eRestrict:
    compiler_type = ref_type->GetFullCompilerType().AddRestrictModifier();
    break;
  default:
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("ParseModifierType called with unsupported kind: {0}",
                      kind),
        llvm::inconvertibleErrorCode());
  }

  Declaration decl;
  return MakeType(uid, ConstString(), 0, nullptr, LLDB_INVALID_UID,
                  Type::eEncodingIsUID, decl, compiler_type,
                  lldb_private::Type::ResolveState::Full);
}

llvm::Expected<lldb::TypeSP> SymbolFileCTF::ParseTypedef(lldb::offset_t &offset,
                                                         lldb::user_id_t uid,
                                                         llvm::StringRef name,
                                                         uint32_t type) {
  TypeSP underlying_type = GetTypeForUID(type);
  if (!underlying_type)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Could not find typedef underlying type: {0}", type),
        llvm::inconvertibleErrorCode());

  CompilerType target_ast_type = underlying_type->GetFullCompilerType();
  clang::DeclContext *decl_ctx = m_ast->GetTranslationUnitDecl();
  CompilerType ast_typedef = target_ast_type.CreateTypedef(
      name.data(), m_ast->CreateDeclContext(decl_ctx), 0);

  Declaration decl;
  return MakeType(uid, ConstString(name), 0, nullptr, LLDB_INVALID_UID,
                  lldb_private::Type::eEncodingIsUID, decl, ast_typedef,
                  lldb_private::Type::ResolveState::Full);
}

llvm::Expected<lldb::TypeSP> SymbolFileCTF::ParseArray(lldb::offset_t &offset,
                                                       lldb::user_id_t uid,
                                                       llvm::StringRef name) {
  ctf_array_t ctf_array;
  ctf_array.contents = m_data.GetU32(&offset);
  ctf_array.index = m_data.GetU32(&offset);
  ctf_array.nelems = m_data.GetU32(&offset);

  TypeSP element_type = GetTypeForUID(ctf_array.contents);
  if (!element_type)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Could not find array element type: {0}",
                      ctf_array.contents),
        llvm::inconvertibleErrorCode());

  std::optional<uint64_t> element_size = element_type->GetByteSize(nullptr);
  if (!element_size)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("could not get element size of type: {0}",
                      ctf_array.contents),
        llvm::inconvertibleErrorCode());

  uint64_t size = ctf_array.nelems * *element_size;

  CompilerType compiler_type = m_ast->CreateArrayType(
      element_type->GetFullCompilerType(), ctf_array.nelems,
      /*is_gnu_vector*/ false);

  Declaration decl;
  return MakeType(uid, ConstString(), size, nullptr, LLDB_INVALID_UID,
                  Type::eEncodingIsUID, decl, compiler_type,
                  lldb_private::Type::ResolveState::Full);
}

llvm::Expected<lldb::TypeSP> SymbolFileCTF::ParseEnum(lldb::offset_t &offset,
                                                      lldb::user_id_t uid,
                                                      llvm::StringRef name,
                                                      uint32_t elements,
                                                      uint32_t size) {
  Declaration decl;
  CompilerType enum_type = m_ast->CreateEnumerationType(
      name, m_ast->GetTranslationUnitDecl(), OptionalClangModuleID(), decl,
      m_ast->GetBasicType(eBasicTypeInt),
      /*is_scoped=*/false);

  for (uint32_t i = 0; i < elements; ++i) {
    ctf_enum_t ctf_enum;
    ctf_enum.name = m_data.GetU32(&offset);
    ctf_enum.value = m_data.GetU32(&offset);

    llvm::StringRef value_name = ReadString(ctf_enum.name);
    const uint32_t value = ctf_enum.value;

    Declaration value_decl;
    m_ast->AddEnumerationValueToEnumerationType(enum_type, value_decl,
                                                value_name.data(), value, size);
  }

  return MakeType(uid, ConstString(), 0, nullptr, LLDB_INVALID_UID,
                  Type::eEncodingIsUID, decl, enum_type,
                  lldb_private::Type::ResolveState::Full);
}

llvm::Expected<lldb::TypeSP>
SymbolFileCTF::ParseFunction(lldb::offset_t &offset, lldb::user_id_t uid,
                             llvm::StringRef name, uint32_t num_args,
                             uint32_t type) {
  std::vector<CompilerType> arg_types;
  arg_types.reserve(num_args);

  bool is_variadic = false;
  for (uint32_t i = 0; i < num_args; ++i) {
    const uint32_t arg_uid = m_data.GetU32(&offset);

    // If the last argument is 0, this is a variadic function.
    if (arg_uid == 0) {
      is_variadic = true;
      break;
    }

    if (TypeSP arg_type = GetTypeForUID(arg_uid))
      arg_types.push_back(arg_type->GetFullCompilerType());
  }

  // If the number of arguments is odd, a single uint32_t of padding is inserted
  // to maintain alignment.
  if (num_args % 2 == 1)
    m_data.GetU32(&offset);

  TypeSP ret_type = GetTypeForUID(type);
  if (!ret_type)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Could not find function return type: {0}", type),
        llvm::inconvertibleErrorCode());

  CompilerType func_type = m_ast->CreateFunctionType(
      ret_type->GetFullCompilerType(), arg_types.data(), arg_types.size(),
      is_variadic, 0, clang::CallingConv::CC_C);

  Declaration decl;
  return MakeType(uid, ConstString(name), 0, nullptr, LLDB_INVALID_UID,
                  Type::eEncodingIsUID, decl, func_type,
                  lldb_private::Type::ResolveState::Full);
}

llvm::Expected<lldb::TypeSP>
SymbolFileCTF::ParseRecord(lldb::offset_t &offset, lldb::user_id_t uid,
                           llvm::StringRef name, uint32_t kind, uint32_t fields,
                           uint32_t size) {
  const clang::TagTypeKind tag_kind =
      TranslateRecordKind(static_cast<TypeKind>(kind));

  CompilerType union_type =
      m_ast->CreateRecordType(nullptr, OptionalClangModuleID(), eAccessPublic,
                              name.data(), tag_kind, eLanguageTypeC);

  m_ast->StartTagDeclarationDefinition(union_type);
  for (uint32_t i = 0; i < fields; ++i) {
    ctf_member_t ctf_member;
    ctf_member.name = m_data.GetU32(&offset);
    ctf_member.type = m_data.GetU32(&offset);
    ctf_member.offset = m_data.GetU16(&offset);
    ctf_member.padding = m_data.GetU16(&offset);

    llvm::StringRef member_name = ReadString(ctf_member.name);
    const uint32_t member_type_uid = ctf_member.type;

    if (TypeSP member_type = GetTypeForUID(member_type_uid)) {
      const uint32_t member_size =
          member_type->GetByteSize(nullptr).value_or(0);
      TypeSystemClang::AddFieldToRecordType(union_type, member_name,
                                            member_type->GetFullCompilerType(),
                                            eAccessPublic, member_size);
    }
  }
  m_ast->CompleteTagDeclarationDefinition(union_type);

  Declaration decl;
  return MakeType(uid, ConstString(name), size, nullptr, LLDB_INVALID_UID,
                  lldb_private::Type::eEncodingIsUID, decl, union_type,
                  lldb_private::Type::ResolveState::Full);
}

llvm::Expected<TypeSP> SymbolFileCTF::ParseType(
    lldb::offset_t &offset, lldb::user_id_t uid, llvm::StringRef name,
    uint32_t kind, uint32_t variable_length, uint32_t type, uint32_t size) {
  switch (kind) {
  case TypeKind::eInteger:
    return ParseInteger(offset, uid, name);
  case TypeKind::eConst:
  case TypeKind::ePointer:
  case TypeKind::eRestrict:
  case TypeKind::eVolatile:
    return ParseModifierType(offset, uid, kind, type);
  case TypeKind::eTypedef:
    return ParseTypedef(offset, uid, name, type);
  case TypeKind::eArray:
    return ParseArray(offset, uid, name);
  case TypeKind::eEnum:
    return ParseEnum(offset, uid, name, variable_length, size);
  case TypeKind::eFunction:
    return ParseFunction(offset, uid, name, variable_length, size);
  case TypeKind::eStruct:
  case TypeKind::eUnion:
    return ParseRecord(offset, uid, name, kind, variable_length, size);
  case TypeKind::eFloat:
  case TypeKind::eForward:
  case TypeKind::eSlice:
  case TypeKind::eUnknown:
    offset += (variable_length * sizeof(uint32_t));
    break;
  }
  return llvm::make_error<llvm::StringError>(
      llvm::formatv("unsupported type (name = {0}, kind = {1}, vlength = {2})",
                    name, kind, variable_length),
      llvm::inconvertibleErrorCode());
}

size_t SymbolFileCTF::ParseTypes(CompileUnit &cu) {
  if (!ParseHeader())
    return 0;

  if (!m_types.empty())
    return 0;

  if (!m_ast)
    return 0;

  Log *log = GetLog(LLDBLog::Symbols);
  LLDB_LOG(log, "Parsing CTF types");

  lldb::offset_t type_offset = m_body_offset + m_header->typeoff;
  const lldb::offset_t type_offset_end = m_body_offset + m_header->stroff;

  lldb::user_id_t type_uid = 1;
  while (type_offset < type_offset_end) {
    ctf_stype_t ctf_stype;
    ctf_stype.name = m_data.GetU32(&type_offset);
    ctf_stype.info = m_data.GetU32(&type_offset);
    ctf_stype.size = m_data.GetU32(&type_offset);

    llvm::StringRef name = ReadString(ctf_stype.name);
    const uint32_t kind = GetKind(ctf_stype.info);
    const uint32_t variable_length = GetVLen(ctf_stype.info);
    const uint32_t type = ctf_stype.GetType();
    const uint32_t size = ctf_stype.GetSize();

    TypeSP type_sp;
    llvm::Expected<TypeSP> type_or_error = ParseType(
        type_offset, type_uid, name, kind, variable_length, type, size);
    if (!type_or_error) {
      LLDB_LOG_ERROR(log, type_or_error.takeError(),
                     "Failed to parse type {1} at offset {2}: {0}", type_uid,
                     type_offset);
    } else {
      type_sp = *type_or_error;
      if (log) {
        StreamString ss;
        type_sp->Dump(&ss, true);
        LLDB_LOGV(log, "Adding type {0}: {1}", type_uid,
                  llvm::StringRef(ss.GetString()).rtrim());
      }
    }

    AddTypeForUID(type_uid++, type_sp);
  }

  if (log) {
    size_t skipped_types = 0;
    for (auto &type : m_types) {
      if (!type)
        skipped_types++;
    }
    LLDB_LOG(log, "Parsed {0} CTF types ({1} skipped)", m_types.size(),
             skipped_types);
  }

  return m_types.size();
}

size_t SymbolFileCTF::ParseFunctions(CompileUnit &cu) {
  if (!ParseHeader())
    return 0;

  if (!m_functions.empty())
    return 0;

  if (!m_ast)
    return 0;

  Symtab *symtab = GetObjectFile()->GetModule()->GetSymtab();
  if (!symtab)
    return 0;

  Log *log = GetLog(LLDBLog::Symbols);
  LLDB_LOG(log, "Parsing CTF functions");

  lldb::offset_t function_offset = m_body_offset + m_header->funcoff;
  const lldb::offset_t function_offset_end = m_body_offset + m_header->typeoff;

  uint32_t symbol_idx = 0;
  Declaration decl;
  while (function_offset < function_offset_end) {
    const uint32_t info = m_data.GetU32(&function_offset);
    const uint16_t kind = GetKind(info);
    const uint16_t variable_length = GetVLen(info);

    Symbol *symbol = symtab->FindSymbolWithType(
        eSymbolTypeCode, Symtab::eDebugYes, Symtab::eVisibilityAny, symbol_idx);

    // Skip padding.
    if (kind == TypeKind::eUnknown && variable_length == 0)
      continue;

    // Skip unexpected kinds.
    if (kind != TypeKind::eFunction)
      continue;

    const uint32_t ret_uid = m_data.GetU32(&function_offset);
    const uint32_t num_args = variable_length;

    std::vector<CompilerType> arg_types;
    arg_types.reserve(num_args);

    bool is_variadic = false;
    for (uint32_t i = 0; i < variable_length; i++) {
      const uint32_t arg_uid = m_data.GetU32(&function_offset);

      // If the last argument is 0, this is a variadic function.
      if (arg_uid == 0) {
        is_variadic = true;
        break;
      }

      TypeSP arg_type = GetTypeForUID(arg_uid);
      arg_types.push_back(arg_type->GetFullCompilerType());
    }

    if (symbol) {
      TypeSP ret_type = GetTypeForUID(ret_uid);
      AddressRange func_range =
          AddressRange(symbol->GetFileAddress(), symbol->GetByteSize(),
                       GetObjectFile()->GetModule()->GetSectionList());

      // Create function type.
      CompilerType func_type = m_ast->CreateFunctionType(
          ret_type->GetFullCompilerType(), arg_types.data(), arg_types.size(),
          is_variadic, 0, clang::CallingConv::CC_C);
      lldb::user_id_t function_type_uid = m_types.size() + 1;
      TypeSP type_sp =
          MakeType(function_type_uid, symbol->GetName(), 0, nullptr,
                   LLDB_INVALID_UID, Type::eEncodingIsUID, decl, func_type,
                   lldb_private::Type::ResolveState::Full);
      AddTypeForUID(function_type_uid, type_sp);

      // Create function.
      lldb::user_id_t func_uid = m_functions.size();
      FunctionSP function_sp = std::make_shared<Function>(
          &cu, func_uid, function_type_uid, symbol->GetMangled(), type_sp.get(),
          func_range);
      m_functions.emplace_back(function_sp);
      cu.AddFunction(function_sp);
    }
  }

  LLDB_LOG(log, "CTF parsed {0} functions", m_functions.size());

  return m_functions.size();
}

static DWARFExpression CreateDWARFExpression(ModuleSP module_sp,
                                             const Symbol &symbol) {
  if (!module_sp)
    return DWARFExpression();

  const ArchSpec &architecture = module_sp->GetArchitecture();
  ByteOrder byte_order = architecture.GetByteOrder();
  uint32_t address_size = architecture.GetAddressByteSize();
  uint32_t byte_size = architecture.GetDataByteSize();

  StreamBuffer<32> stream(Stream::eBinary, address_size, byte_order);
  stream.PutHex8(lldb_private::dwarf::DW_OP_addr);
  stream.PutMaxHex64(symbol.GetFileAddress(), address_size, byte_order);

  DataBufferSP buffer =
      std::make_shared<DataBufferHeap>(stream.GetData(), stream.GetSize());
  lldb_private::DataExtractor extractor(buffer, byte_order, address_size,
                                        byte_size);
  DWARFExpression result(extractor);
  result.SetRegisterKind(eRegisterKindDWARF);

  return result;
}

size_t SymbolFileCTF::ParseObjects(CompileUnit &comp_unit) {
  if (!ParseHeader())
    return 0;

  if (!m_variables.empty())
    return 0;

  if (!m_ast)
    return 0;

  ModuleSP module_sp = GetObjectFile()->GetModule();
  Symtab *symtab = module_sp->GetSymtab();
  if (!symtab)
    return 0;

  Log *log = GetLog(LLDBLog::Symbols);
  LLDB_LOG(log, "Parsing CTF objects");

  lldb::offset_t object_offset = m_body_offset + m_header->objtoff;
  const lldb::offset_t object_offset_end = m_body_offset + m_header->funcoff;

  uint32_t symbol_idx = 0;
  Declaration decl;
  while (object_offset < object_offset_end) {
    const uint32_t type_uid = m_data.GetU32(&object_offset);

    if (Symbol *symbol =
            symtab->FindSymbolWithType(eSymbolTypeData, Symtab::eDebugYes,
                                       Symtab::eVisibilityAny, symbol_idx)) {

      Variable::RangeList ranges;
      ranges.Append(symbol->GetFileAddress(), symbol->GetByteSize());

      auto type_sp = std::make_shared<SymbolFileType>(*this, type_uid);

      DWARFExpressionList location(
          module_sp, CreateDWARFExpression(module_sp, *symbol), nullptr);

      lldb::user_id_t variable_type_uid = m_variables.size();
      m_variables.emplace_back(std::make_shared<Variable>(
          variable_type_uid, symbol->GetName().AsCString(),
          symbol->GetName().AsCString(), type_sp, eValueTypeVariableGlobal,
          m_comp_unit_sp.get(), ranges, &decl, location, symbol->IsExternal(),
          /*artificial=*/false,
          /*location_is_constant_data*/ false));
    }
  }

  LLDB_LOG(log, "Parsed {0} CTF objects", m_variables.size());

  return m_variables.size();
}

uint32_t SymbolFileCTF::CalculateAbilities() {
  if (!m_objfile_sp)
    return 0;

  if (!ParseHeader())
    return 0;

  return VariableTypes | Functions | GlobalVariables;
}

uint32_t SymbolFileCTF::ResolveSymbolContext(const Address &so_addr,
                                             SymbolContextItem resolve_scope,
                                             SymbolContext &sc) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  if (m_objfile_sp->GetSymtab() == nullptr)
    return 0;

  uint32_t resolved_flags = 0;

  // Resolve symbols.
  if (resolve_scope & eSymbolContextSymbol) {
    sc.symbol = m_objfile_sp->GetSymtab()->FindSymbolContainingFileAddress(
        so_addr.GetFileAddress());
    if (sc.symbol)
      resolved_flags |= eSymbolContextSymbol;
  }

  // Resolve functions.
  if (resolve_scope & eSymbolContextFunction) {
    for (FunctionSP function_sp : m_functions) {
      if (function_sp->GetAddressRange().ContainsFileAddress(
              so_addr.GetFileAddress())) {
        sc.function = function_sp.get();
        resolved_flags |= eSymbolContextFunction;
        break;
      }
    }
  }

  // Resolve variables.
  if (resolve_scope & eSymbolContextVariable) {
    for (VariableSP variable_sp : m_variables) {
      if (variable_sp->LocationIsValidForAddress(so_addr.GetFileAddress())) {
        sc.variable = variable_sp.get();
        break;
      }
    }
  }

  return resolved_flags;
}

CompUnitSP SymbolFileCTF::ParseCompileUnitAtIndex(uint32_t idx) {
  if (idx == 0)
    return m_comp_unit_sp;
  return {};
}

size_t
SymbolFileCTF::ParseVariablesForContext(const lldb_private::SymbolContext &sc) {
  return ParseObjects(*m_comp_unit_sp);
}

void SymbolFileCTF::AddSymbols(Symtab &symtab) {
  // CTF does not encode symbols.
  // We rely on the existing symbol table to map symbols to type.
}

void SymbolFileCTF::AddTypeForUID(lldb::user_id_t type_uid, lldb::TypeSP type) {
  assert(type_uid == m_types.size() + 1);
  m_types.emplace_back(type);
}

TypeSP SymbolFileCTF::GetTypeForUID(lldb::user_id_t type_uid) {
  if (type_uid > m_types.size())
    return {};

  if (type_uid < 1)
    return {};

  return m_types[type_uid - 1];
}

lldb_private::Type *SymbolFileCTF::ResolveTypeUID(lldb::user_id_t type_uid) {
  return GetTypeForUID(type_uid).get();
}

void SymbolFileCTF::FindTypes(
    lldb_private::ConstString name,
    const lldb_private::CompilerDeclContext &parent_decl_ctx,
    uint32_t max_matches,
    llvm::DenseSet<lldb_private::SymbolFile *> &searched_symbol_files,
    lldb_private::TypeMap &types) {

  searched_symbol_files.clear();
  searched_symbol_files.insert(this);

  size_t matches = 0;
  for (TypeSP type_sp : m_types) {
    if (matches == max_matches)
      break;
    if (type_sp && type_sp->GetName() == name) {
      types.Insert(type_sp);
      matches++;
    }
  }
}

void SymbolFileCTF::FindTypesByRegex(
    const lldb_private::RegularExpression &regex, uint32_t max_matches,
    lldb_private::TypeMap &types) {
  ParseTypes(*m_comp_unit_sp);

  size_t matches = 0;
  for (TypeSP type_sp : m_types) {
    if (matches == max_matches)
      break;
    if (type_sp && regex.Execute(type_sp->GetName()))
      types.Insert(type_sp);
    matches++;
  }
}

void SymbolFileCTF::FindFunctions(
    const lldb_private::Module::LookupInfo &lookup_info,
    const lldb_private::CompilerDeclContext &parent_decl_ctx,
    bool include_inlines, lldb_private::SymbolContextList &sc_list) {
  ParseFunctions(*m_comp_unit_sp);

  ConstString name = lookup_info.GetLookupName();
  for (FunctionSP function_sp : m_functions) {
    if (function_sp && function_sp->GetName() == name) {
      lldb_private::SymbolContext sc;
      sc.comp_unit = m_comp_unit_sp.get();
      sc.function = function_sp.get();
      sc_list.Append(sc);
    }
  }
}

void SymbolFileCTF::FindFunctions(const lldb_private::RegularExpression &regex,
                                  bool include_inlines,
                                  lldb_private::SymbolContextList &sc_list) {
  for (FunctionSP function_sp : m_functions) {
    if (function_sp && regex.Execute(function_sp->GetName())) {
      lldb_private::SymbolContext sc;
      sc.comp_unit = m_comp_unit_sp.get();
      sc.function = function_sp.get();
      sc_list.Append(sc);
    }
  }
}

void SymbolFileCTF::FindGlobalVariables(
    lldb_private::ConstString name,
    const lldb_private::CompilerDeclContext &parent_decl_ctx,
    uint32_t max_matches, lldb_private::VariableList &variables) {
  ParseObjects(*m_comp_unit_sp);

  size_t matches = 0;
  for (VariableSP variable_sp : m_variables) {
    if (matches == max_matches)
      break;
    if (variable_sp && variable_sp->GetName() == name) {
      variables.AddVariable(variable_sp);
      matches++;
    }
  }
}

void SymbolFileCTF::FindGlobalVariables(
    const lldb_private::RegularExpression &regex, uint32_t max_matches,
    lldb_private::VariableList &variables) {
  ParseObjects(*m_comp_unit_sp);

  size_t matches = 0;
  for (VariableSP variable_sp : m_variables) {
    if (matches == max_matches)
      break;
    if (variable_sp && regex.Execute(variable_sp->GetName())) {
      variables.AddVariable(variable_sp);
      matches++;
    }
  }
}

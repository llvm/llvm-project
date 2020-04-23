//===-- Type.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#include "lldb/Core/Module.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/StreamString.h"

#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContextScope.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/TypeSystem.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;

bool lldb_private::contextMatches(llvm::ArrayRef<CompilerContext> context_chain,
                                  llvm::ArrayRef<CompilerContext> pattern) {
  auto ctx = context_chain.begin();
  auto ctx_end = context_chain.end();
  for (const CompilerContext &pat : pattern) {
    // Early exit if the pattern is too long.
    if (ctx == ctx_end)
      return false;
    if (*ctx != pat) {
      // Skip any number of module matches.
      if (pat.kind == CompilerContextKind::AnyModule) {
        // Greedily match 0..n modules.
        ctx = std::find_if(ctx, ctx_end, [](const CompilerContext &ctx) {
          return ctx.kind != CompilerContextKind::Module;
        });
        continue;
      }
      // See if there is a kind mismatch; they should have 1 bit in common.
      if (((uint16_t)ctx->kind & (uint16_t)pat.kind) == 0)
        return false;
      // The name is ignored for AnyModule, but not for AnyType.
      if (pat.kind != CompilerContextKind::AnyModule && ctx->name != pat.name)
        return false;
    }
    ++ctx;
  }
  return true;
}

void CompilerContext::Dump() const {
  switch (kind) {
  default:
    printf("Invalid");
    break;
  case CompilerContextKind::TranslationUnit:
    printf("TranslationUnit");
    break;
  case CompilerContextKind::Module:
    printf("Module");
    break;
  case CompilerContextKind::Namespace:
    printf("Namespace");
    break;
  case CompilerContextKind::Class:
    printf("Class");
    break;
  case CompilerContextKind::Struct:
    printf("Structure");
    break;
  case CompilerContextKind::Union:
    printf("Union");
    break;
  case CompilerContextKind::Function:
    printf("Function");
    break;
  case CompilerContextKind::Variable:
    printf("Variable");
    break;
  case CompilerContextKind::Enum:
    printf("Enumeration");
    break;
  case CompilerContextKind::Typedef:
    printf("Typedef");
    break;
  case CompilerContextKind::AnyModule:
    printf("AnyModule");
    break;
  case CompilerContextKind::AnyType:
    printf("AnyType");
    break;
  }
  printf("(\"%s\")\n", name.GetCString());
}

class TypeAppendVisitor {
public:
  TypeAppendVisitor(TypeListImpl &type_list) : m_type_list(type_list) {}

  bool operator()(const lldb::TypeSP &type) {
    m_type_list.Append(TypeImplSP(new TypeImpl(type)));
    return true;
  }

private:
  TypeListImpl &m_type_list;
};

void TypeListImpl::Append(const lldb_private::TypeList &type_list) {
  TypeAppendVisitor cb(*this);
  type_list.ForEach(cb);
}

SymbolFileType::SymbolFileType(SymbolFile &symbol_file,
                               const lldb::TypeSP &type_sp)
    : UserID(type_sp ? type_sp->GetID() : LLDB_INVALID_UID),
      m_symbol_file(symbol_file), m_type_sp(type_sp) {}

Type *SymbolFileType::GetType() {
  if (!m_type_sp) {
    Type *resolved_type = m_symbol_file.ResolveTypeUID(GetID());
    if (resolved_type)
      m_type_sp = resolved_type->shared_from_this();
  }
  return m_type_sp.get();
}

Type::Type(lldb::user_id_t uid, SymbolFile *symbol_file, ConstString name,
           llvm::Optional<uint64_t> byte_size, SymbolContextScope *context,
           user_id_t encoding_uid, EncodingDataType encoding_uid_type,
           const Declaration &decl, const CompilerType &compiler_type,
           ResolveState compiler_type_resolve_state, uint32_t opaque_payload)
    : std::enable_shared_from_this<Type>(), UserID(uid), m_name(name),
      m_symbol_file(symbol_file), m_context(context), m_encoding_type(nullptr),
      m_encoding_uid(encoding_uid), m_encoding_uid_type(encoding_uid_type),
      m_decl(decl), m_compiler_type(compiler_type),
      m_compiler_type_resolve_state(compiler_type ? compiler_type_resolve_state
                                                  : ResolveState::Unresolved),
      m_payload(opaque_payload) {
  if (byte_size) {
    m_byte_size = *byte_size;
    m_byte_size_has_value = true;
  } else {
    m_byte_size = 0;
    m_byte_size_has_value = false;
  }
}

Type::Type()
    : std::enable_shared_from_this<Type>(), UserID(0), m_name("<INVALID TYPE>"),
      m_symbol_file(nullptr), m_context(nullptr), m_encoding_type(nullptr),
      m_encoding_uid(LLDB_INVALID_UID), m_encoding_uid_type(eEncodingInvalid),
      m_compiler_type_resolve_state(ResolveState::Unresolved) {
  m_byte_size = 0;
  m_byte_size_has_value = false;
}

void Type::GetDescription(Stream *s, lldb::DescriptionLevel level,
                          bool show_name) {
  *s << "id = " << (const UserID &)*this;

  // Call the name accessor to make sure we resolve the type name
  if (show_name) {
    ConstString type_name = GetName();
    if (type_name) {
      *s << ", name = \"" << type_name << '"';
      ConstString qualified_type_name(GetQualifiedName());
      if (qualified_type_name != type_name) {
        *s << ", qualified = \"" << qualified_type_name << '"';
      }
    }
  }

  // Call the get byte size accesor so we resolve our byte size
  if (GetByteSize())
    s->Printf(", byte-size = %" PRIu64, m_byte_size);
  bool show_fullpaths = (level == lldb::eDescriptionLevelVerbose);
  m_decl.Dump(s, show_fullpaths);

  if (m_compiler_type.IsValid()) {
    *s << ", compiler_type = \"";
    GetForwardCompilerType().DumpTypeDescription(s);
    *s << '"';
  } else if (m_encoding_uid != LLDB_INVALID_UID) {
    s->Printf(", type_uid = 0x%8.8" PRIx64, m_encoding_uid);
    switch (m_encoding_uid_type) {
    case eEncodingInvalid:
      break;
    case eEncodingIsUID:
      s->PutCString(" (unresolved type)");
      break;
    case eEncodingIsConstUID:
      s->PutCString(" (unresolved const type)");
      break;
    case eEncodingIsRestrictUID:
      s->PutCString(" (unresolved restrict type)");
      break;
    case eEncodingIsVolatileUID:
      s->PutCString(" (unresolved volatile type)");
      break;
    case eEncodingIsAtomicUID:
      s->PutCString(" (unresolved atomic type)");
      break;
    case eEncodingIsTypedefUID:
      s->PutCString(" (unresolved typedef)");
      break;
    case eEncodingIsPointerUID:
      s->PutCString(" (unresolved pointer)");
      break;
    case eEncodingIsLValueReferenceUID:
      s->PutCString(" (unresolved L value reference)");
      break;
    case eEncodingIsRValueReferenceUID:
      s->PutCString(" (unresolved R value reference)");
      break;
    case eEncodingIsSyntheticUID:
      s->PutCString(" (synthetic type)");
      break;
    }
  }
}

void Type::Dump(Stream *s, bool show_context, lldb::DescriptionLevel level) {
  s->Printf("%p: ", static_cast<void *>(this));
  s->Indent();
  *s << "Type" << static_cast<const UserID &>(*this) << ' ';
  if (m_name)
    *s << ", name = \"" << m_name << "\"";

  if (m_byte_size_has_value)
    s->Printf(", size = %" PRIu64, m_byte_size);

  if (show_context && m_context != nullptr) {
    s->PutCString(", context = ( ");
    m_context->DumpSymbolContext(s);
    s->PutCString(" )");
  }

  bool show_fullpaths = false;
  m_decl.Dump(s, show_fullpaths);

  if (m_compiler_type.IsValid()) {
    *s << ", compiler_type = " << m_compiler_type.GetOpaqueQualType() << ' ';
    GetForwardCompilerType().DumpTypeDescription(s, level);
  } else if (m_encoding_uid != LLDB_INVALID_UID) {
    s->Format(", type_data = {0:x-16}", m_encoding_uid);
    switch (m_encoding_uid_type) {
    case eEncodingInvalid:
      break;
    case eEncodingIsUID:
      s->PutCString(" (unresolved type)");
      break;
    case eEncodingIsConstUID:
      s->PutCString(" (unresolved const type)");
      break;
    case eEncodingIsRestrictUID:
      s->PutCString(" (unresolved restrict type)");
      break;
    case eEncodingIsVolatileUID:
      s->PutCString(" (unresolved volatile type)");
      break;
    case eEncodingIsAtomicUID:
      s->PutCString(" (unresolved atomic type)");
      break;
    case eEncodingIsTypedefUID:
      s->PutCString(" (unresolved typedef)");
      break;
    case eEncodingIsPointerUID:
      s->PutCString(" (unresolved pointer)");
      break;
    case eEncodingIsLValueReferenceUID:
      s->PutCString(" (unresolved L value reference)");
      break;
    case eEncodingIsRValueReferenceUID:
      s->PutCString(" (unresolved R value reference)");
      break;
    case eEncodingIsSyntheticUID:
      s->PutCString(" (synthetic type)");
      break;
    }
  }

  //
  //  if (m_access)
  //      s->Printf(", access = %u", m_access);
  s->EOL();
}

ConstString Type::GetName() {
  if (!m_name)
    m_name = GetForwardCompilerType().GetTypeName();
  return m_name;
}

void Type::DumpTypeName(Stream *s) { GetName().Dump(s, "<invalid-type-name>"); }

void Type::DumpValue(ExecutionContext *exe_ctx, Stream *s,
                     const DataExtractor &data, uint32_t data_byte_offset,
                     bool show_types, bool show_summary, bool verbose,
                     lldb::Format format) {
  if (ResolveCompilerType(ResolveState::Forward)) {
    if (show_types) {
      s->PutChar('(');
      if (verbose)
        s->Printf("Type{0x%8.8" PRIx64 "} ", GetID());
      DumpTypeName(s);
      s->PutCString(") ");
    }

    GetForwardCompilerType().DumpValue(
        exe_ctx, s, format == lldb::eFormatDefault ? GetFormat() : format, data,
        data_byte_offset, GetByteSize().getValueOr(0),
        0, // Bitfield bit size
        0, // Bitfield bit offset
        show_types, show_summary, verbose, 0);
  }
}

Type *Type::GetEncodingType() {
  if (m_encoding_type == nullptr && m_encoding_uid != LLDB_INVALID_UID)
    m_encoding_type = m_symbol_file->ResolveTypeUID(m_encoding_uid);
  return m_encoding_type;
}

llvm::Optional<uint64_t> Type::GetByteSize() {
  if (m_byte_size_has_value)
    return m_byte_size;

  switch (m_encoding_uid_type) {
  case eEncodingInvalid:
  case eEncodingIsSyntheticUID:
    break;
  case eEncodingIsUID:
  case eEncodingIsConstUID:
  case eEncodingIsRestrictUID:
  case eEncodingIsVolatileUID:
  case eEncodingIsAtomicUID:
  case eEncodingIsTypedefUID: {
    Type *encoding_type = GetEncodingType();
    if (encoding_type)
      if (llvm::Optional<uint64_t> size = encoding_type->GetByteSize()) {
        m_byte_size = *size;
        m_byte_size_has_value = true;
        return m_byte_size;
      }

    if (llvm::Optional<uint64_t> size =
            GetLayoutCompilerType().GetByteSize(nullptr)) {
      m_byte_size = *size;
      m_byte_size_has_value = true;
        return m_byte_size;
    }
  } break;

    // If we are a pointer or reference, then this is just a pointer size;
    case eEncodingIsPointerUID:
    case eEncodingIsLValueReferenceUID:
    case eEncodingIsRValueReferenceUID: {
      if (ArchSpec arch = m_symbol_file->GetObjectFile()->GetArchitecture()) {
        m_byte_size = arch.GetAddressByteSize();
        m_byte_size_has_value = true;
      }
    } break;
  }
  return {};
}

uint32_t Type::GetNumChildren(bool omit_empty_base_classes) {
  return GetForwardCompilerType().GetNumChildren(omit_empty_base_classes, nullptr);
}

bool Type::IsAggregateType() {
  return GetForwardCompilerType().IsAggregateType();
}

lldb::TypeSP Type::GetTypedefType() {
  lldb::TypeSP type_sp;
  if (IsTypedef()) {
    Type *typedef_type = m_symbol_file->ResolveTypeUID(m_encoding_uid);
    if (typedef_type)
      type_sp = typedef_type->shared_from_this();
  }
  return type_sp;
}

lldb::Format Type::GetFormat() { return GetForwardCompilerType().GetFormat(); }

lldb::Encoding Type::GetEncoding(uint64_t &count) {
  // Make sure we resolve our type if it already hasn't been.
  return GetForwardCompilerType().GetEncoding(count);
}

bool Type::DumpValueInMemory(ExecutionContext *exe_ctx, Stream *s,
                             lldb::addr_t address, AddressType address_type,
                             bool show_types, bool show_summary, bool verbose) {
  if (address != LLDB_INVALID_ADDRESS) {
    DataExtractor data;
    Target *target = nullptr;
    if (exe_ctx)
      target = exe_ctx->GetTargetPtr();
    if (target)
      data.SetByteOrder(target->GetArchitecture().GetByteOrder());
    if (ReadFromMemory(exe_ctx, address, address_type, data)) {
      DumpValue(exe_ctx, s, data, 0, show_types, show_summary, verbose);
      return true;
    }
  }
  return false;
}

bool Type::ReadFromMemory(ExecutionContext *exe_ctx, lldb::addr_t addr,
                          AddressType address_type, DataExtractor &data) {
  if (address_type == eAddressTypeFile) {
    // Can't convert a file address to anything valid without more context
    // (which Module it came from)
    return false;
  }

  const uint64_t byte_size = GetByteSize().getValueOr(0);
  if (data.GetByteSize() < byte_size) {
    lldb::DataBufferSP data_sp(new DataBufferHeap(byte_size, '\0'));
    data.SetData(data_sp);
  }

  uint8_t *dst = const_cast<uint8_t *>(data.PeekData(0, byte_size));
  if (dst != nullptr) {
    if (address_type == eAddressTypeHost) {
      // The address is an address in this process, so just copy it
      if (addr == 0)
        return false;
      memcpy(dst, reinterpret_cast<uint8_t *>(addr), byte_size);
      return true;
    } else {
      if (exe_ctx) {
        Process *process = exe_ctx->GetProcessPtr();
        if (process) {
          Status error;
          return exe_ctx->GetProcessPtr()->ReadMemory(addr, dst, byte_size,
                                                      error) == byte_size;
        }
      }
    }
  }
  return false;
}

bool Type::WriteToMemory(ExecutionContext *exe_ctx, lldb::addr_t addr,
                         AddressType address_type, DataExtractor &data) {
  return false;
}

const Declaration &Type::GetDeclaration() const { return m_decl; }

bool Type::ResolveCompilerType(ResolveState compiler_type_resolve_state) {
  // TODO: This needs to consider the correct type system to use.
  Type *encoding_type = nullptr;
  if (!m_compiler_type.IsValid()) {
    encoding_type = GetEncodingType();
    if (encoding_type) {
      switch (m_encoding_uid_type) {
      case eEncodingIsUID: {
        CompilerType encoding_compiler_type =
            encoding_type->GetForwardCompilerType();
        if (encoding_compiler_type.IsValid()) {
          m_compiler_type = encoding_compiler_type;
          m_compiler_type_resolve_state =
              encoding_type->m_compiler_type_resolve_state;
        }
      } break;

      case eEncodingIsConstUID:
        m_compiler_type =
            encoding_type->GetForwardCompilerType().AddConstModifier();
        break;

      case eEncodingIsRestrictUID:
        m_compiler_type =
            encoding_type->GetForwardCompilerType().AddRestrictModifier();
        break;

      case eEncodingIsVolatileUID:
        m_compiler_type =
            encoding_type->GetForwardCompilerType().AddVolatileModifier();
        break;

      case eEncodingIsAtomicUID:
        m_compiler_type =
            encoding_type->GetForwardCompilerType().GetAtomicType();
        break;

      case eEncodingIsTypedefUID:
        m_compiler_type = encoding_type->GetForwardCompilerType().CreateTypedef(
            m_name.AsCString("__lldb_invalid_typedef_name"),
            GetSymbolFile()->GetDeclContextContainingUID(GetID()), m_payload);
        m_name.Clear();
        break;

      case eEncodingIsPointerUID:
        m_compiler_type =
            encoding_type->GetForwardCompilerType().GetPointerType();
        break;

      case eEncodingIsLValueReferenceUID:
        m_compiler_type =
            encoding_type->GetForwardCompilerType().GetLValueReferenceType();
        break;

      case eEncodingIsRValueReferenceUID:
        m_compiler_type =
            encoding_type->GetForwardCompilerType().GetRValueReferenceType();
        break;

      default:
        llvm_unreachable("Unhandled encoding_data_type.");
      }
    } else {
      // We have no encoding type, return void?
      auto type_system_or_err =
          m_symbol_file->GetTypeSystemForLanguage(eLanguageTypeC);
      if (auto err = type_system_or_err.takeError()) {
        LLDB_LOG_ERROR(
            lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_SYMBOLS),
            std::move(err),
            "Unable to construct void type from TypeSystemClang");
      } else {
        CompilerType void_compiler_type =
            type_system_or_err->GetBasicTypeFromAST(eBasicTypeVoid);
        switch (m_encoding_uid_type) {
        case eEncodingIsUID:
          m_compiler_type = void_compiler_type;
          break;

        case eEncodingIsConstUID:
          m_compiler_type = void_compiler_type.AddConstModifier();
          break;

        case eEncodingIsRestrictUID:
          m_compiler_type = void_compiler_type.AddRestrictModifier();
          break;

        case eEncodingIsVolatileUID:
          m_compiler_type = void_compiler_type.AddVolatileModifier();
          break;

        case eEncodingIsAtomicUID:
          m_compiler_type = void_compiler_type.GetAtomicType();
          break;

        case eEncodingIsTypedefUID:
          m_compiler_type = void_compiler_type.CreateTypedef(
              m_name.AsCString("__lldb_invalid_typedef_name"),
              GetSymbolFile()->GetDeclContextContainingUID(GetID()), m_payload);
          break;

        case eEncodingIsPointerUID:
          m_compiler_type = void_compiler_type.GetPointerType();
          break;

        case eEncodingIsLValueReferenceUID:
          m_compiler_type = void_compiler_type.GetLValueReferenceType();
          break;

        case eEncodingIsRValueReferenceUID:
          m_compiler_type = void_compiler_type.GetRValueReferenceType();
          break;

        default:
          llvm_unreachable("Unhandled encoding_data_type.");
        }
      }
    }

    // When we have a EncodingUID, our "m_flags.compiler_type_resolve_state" is
    // set to eResolveStateUnresolved so we need to update it to say that we
    // now have a forward declaration since that is what we created above.
    if (m_compiler_type.IsValid())
      m_compiler_type_resolve_state = ResolveState::Forward;
  }

  // Check if we have a forward reference to a class/struct/union/enum?
  if (compiler_type_resolve_state == ResolveState::Layout ||
      compiler_type_resolve_state == ResolveState::Full) {
    // Check if we have a forward reference to a class/struct/union/enum?
    if (m_compiler_type.IsValid() &&
        m_compiler_type_resolve_state < compiler_type_resolve_state) {
      m_compiler_type_resolve_state = ResolveState::Full;
      if (!m_compiler_type.IsDefined()) {
        // We have a forward declaration, we need to resolve it to a complete
        // definition.
        m_symbol_file->CompleteType(m_compiler_type);
      }
    }
  }

  // If we have an encoding type, then we need to make sure it is resolved
  // appropriately.
  if (m_encoding_uid != LLDB_INVALID_UID) {
    if (encoding_type == nullptr)
      encoding_type = GetEncodingType();
    if (encoding_type) {
      ResolveState encoding_compiler_type_resolve_state =
          compiler_type_resolve_state;

      if (compiler_type_resolve_state == ResolveState::Layout) {
        switch (m_encoding_uid_type) {
        case eEncodingIsPointerUID:
        case eEncodingIsLValueReferenceUID:
        case eEncodingIsRValueReferenceUID:
          encoding_compiler_type_resolve_state = ResolveState::Forward;
          break;
        default:
          break;
        }
      }
      encoding_type->ResolveCompilerType(encoding_compiler_type_resolve_state);
    }
  }
  return m_compiler_type.IsValid();
}
uint32_t Type::GetEncodingMask() {
  uint32_t encoding_mask = 1u << m_encoding_uid_type;
  Type *encoding_type = GetEncodingType();
  assert(encoding_type != this);
  if (encoding_type)
    encoding_mask |= encoding_type->GetEncodingMask();
  return encoding_mask;
}

CompilerType Type::GetFullCompilerType() {
  ResolveCompilerType(ResolveState::Full);
  return m_compiler_type;
}

CompilerType Type::GetLayoutCompilerType() {
  ResolveCompilerType(ResolveState::Layout);
  return m_compiler_type;
}

CompilerType Type::GetForwardCompilerType() {
  ResolveCompilerType(ResolveState::Forward);
  return m_compiler_type;
}

ConstString Type::GetQualifiedName() {
  return GetForwardCompilerType().GetTypeName();
}

bool Type::GetTypeScopeAndBasename(const llvm::StringRef& name,
                                   llvm::StringRef &scope,
                                   llvm::StringRef &basename,
                                   TypeClass &type_class) {
  type_class = eTypeClassAny;

  if (name.empty())
    return false;

  basename = name;
  if (basename.consume_front("struct "))
    type_class = eTypeClassStruct;
  else if (basename.consume_front("class "))
    type_class = eTypeClassClass;
  else if (basename.consume_front("union "))
    type_class = eTypeClassUnion;
  else if (basename.consume_front("enum "))
    type_class = eTypeClassEnumeration;
  else if (basename.consume_front("typedef "))
    type_class = eTypeClassTypedef;

  size_t namespace_separator = basename.find("::");
  if (namespace_separator == llvm::StringRef::npos)
    return false;

  size_t template_begin = basename.find('<');
  while (namespace_separator != llvm::StringRef::npos) {
    if (template_begin != llvm::StringRef::npos &&
        namespace_separator > template_begin) {
      size_t template_depth = 1;
      llvm::StringRef template_arg =
          basename.drop_front(template_begin + 1);
      while (template_depth > 0 && !template_arg.empty()) {
        if (template_arg.front() == '<')
          template_depth++;
        else if (template_arg.front() == '>')
          template_depth--;
        template_arg = template_arg.drop_front(1);
      }
      if (template_depth != 0)
        return false; // We have an invalid type name. Bail out.
      if (template_arg.empty())
        break; // The template ends at the end of the full name.
      basename = template_arg;
    } else {
      basename = basename.drop_front(namespace_separator + 2);
    }
    template_begin = basename.find('<');
    namespace_separator = basename.find("::");
  }
  if (basename.size() < name.size()) {
    scope = name.take_front(name.size() - basename.size());
    return true;
  }
  return false;
}

ModuleSP Type::GetModule() {
  if (m_symbol_file)
    return m_symbol_file->GetObjectFile()->GetModule();
  return ModuleSP();
}

TypeAndOrName::TypeAndOrName(TypeSP &in_type_sp) {
  if (in_type_sp) {
    m_compiler_type = in_type_sp->GetForwardCompilerType();
    m_type_name = in_type_sp->GetName();
  }
}

TypeAndOrName::TypeAndOrName(const char *in_type_str)
    : m_type_name(in_type_str) {}

TypeAndOrName::TypeAndOrName(ConstString &in_type_const_string)
    : m_type_name(in_type_const_string) {}

bool TypeAndOrName::operator==(const TypeAndOrName &other) const {
  if (m_compiler_type != other.m_compiler_type)
    return false;
  if (m_type_name != other.m_type_name)
    return false;
  return true;
}

bool TypeAndOrName::operator!=(const TypeAndOrName &other) const {
  return !(*this == other);
}

ConstString TypeAndOrName::GetName() const {
  if (m_type_name)
    return m_type_name;
  if (m_compiler_type)
    return m_compiler_type.GetTypeName();
  return ConstString("<invalid>");
}

void TypeAndOrName::SetName(ConstString type_name) {
  m_type_name = type_name;
}

void TypeAndOrName::SetName(const char *type_name_cstr) {
  m_type_name.SetCString(type_name_cstr);
}

void TypeAndOrName::SetTypeSP(lldb::TypeSP type_sp) {
  if (type_sp) {
    m_compiler_type = type_sp->GetForwardCompilerType();
    m_type_name = type_sp->GetName();
  } else
    Clear();
}

void TypeAndOrName::SetCompilerType(CompilerType compiler_type) {
  m_compiler_type = compiler_type;
  if (m_compiler_type)
    m_type_name = m_compiler_type.GetTypeName();
}

bool TypeAndOrName::IsEmpty() const {
  return !((bool)m_type_name || (bool)m_compiler_type);
}

void TypeAndOrName::Clear() {
  m_type_name.Clear();
  m_compiler_type.Clear();
}

bool TypeAndOrName::HasName() const { return (bool)m_type_name; }

bool TypeAndOrName::HasCompilerType() const {
  return m_compiler_type.IsValid();
}

TypeImpl::TypeImpl(const lldb::TypeSP &type_sp)
    : m_module_wp(), m_static_type(), m_dynamic_type() {
  SetType(type_sp);
}

TypeImpl::TypeImpl(const CompilerType &compiler_type)
    : m_module_wp(), m_static_type(), m_dynamic_type() {
  SetType(compiler_type);
}

TypeImpl::TypeImpl(const lldb::TypeSP &type_sp, const CompilerType &dynamic)
    : m_module_wp(), m_static_type(), m_dynamic_type(dynamic) {
  SetType(type_sp, dynamic);
}

TypeImpl::TypeImpl(const CompilerType &static_type,
                   const CompilerType &dynamic_type)
    : m_module_wp(), m_static_type(), m_dynamic_type() {
  SetType(static_type, dynamic_type);
}

void TypeImpl::SetType(const lldb::TypeSP &type_sp) {
  if (type_sp) {
    m_static_type = type_sp->GetForwardCompilerType();
    m_module_wp = type_sp->GetModule();
  } else {
    m_static_type.Clear();
    m_module_wp = lldb::ModuleWP();
  }
}

void TypeImpl::SetType(const CompilerType &compiler_type) {
  m_module_wp = lldb::ModuleWP();
  m_static_type = compiler_type;
}

void TypeImpl::SetType(const lldb::TypeSP &type_sp,
                       const CompilerType &dynamic) {
  SetType(type_sp);
  m_dynamic_type = dynamic;
}

void TypeImpl::SetType(const CompilerType &compiler_type,
                       const CompilerType &dynamic) {
  m_module_wp = lldb::ModuleWP();
  m_static_type = compiler_type;
  m_dynamic_type = dynamic;
}

bool TypeImpl::CheckModule(lldb::ModuleSP &module_sp) const {
  // Check if we have a module for this type. If we do and the shared pointer
  // is can be successfully initialized with m_module_wp, return true. Else
  // return false if we didn't have a module, or if we had a module and it has
  // been deleted. Any functions doing anything with a TypeSP in this TypeImpl
  // class should call this function and only do anything with the ivars if
  // this function returns true. If we have a module, the "module_sp" will be
  // filled in with a strong reference to the module so that the module will at
  // least stay around long enough for the type query to succeed.
  module_sp = m_module_wp.lock();
  if (!module_sp) {
    lldb::ModuleWP empty_module_wp;
    // If either call to "std::weak_ptr::owner_before(...) value returns true,
    // this indicates that m_module_wp once contained (possibly still does) a
    // reference to a valid shared pointer. This helps us know if we had a
    // valid reference to a section which is now invalid because the module it
    // was in was deleted
    if (empty_module_wp.owner_before(m_module_wp) ||
        m_module_wp.owner_before(empty_module_wp)) {
      // m_module_wp had a valid reference to a module, but all strong
      // references have been released and the module has been deleted
      return false;
    }
  }
  // We either successfully locked the module, or didn't have one to begin with
  return true;
}

bool TypeImpl::operator==(const TypeImpl &rhs) const {
  return m_static_type == rhs.m_static_type &&
         m_dynamic_type == rhs.m_dynamic_type;
}

bool TypeImpl::operator!=(const TypeImpl &rhs) const {
  return !(*this == rhs);
}

bool TypeImpl::IsValid() const {
  // just a name is not valid
  ModuleSP module_sp;
  if (CheckModule(module_sp))
    return m_static_type.IsValid() || m_dynamic_type.IsValid();
  return false;
}

TypeImpl::operator bool() const { return IsValid(); }

void TypeImpl::Clear() {
  m_module_wp = lldb::ModuleWP();
  m_static_type.Clear();
  m_dynamic_type.Clear();
}

ConstString TypeImpl::GetName() const {
  ModuleSP module_sp;
  if (CheckModule(module_sp)) {
    if (m_dynamic_type)
      return m_dynamic_type.GetTypeName();
    return m_static_type.GetTypeName();
  }
  return ConstString();
}

ConstString TypeImpl::GetDisplayTypeName() const {
  ModuleSP module_sp;
  if (CheckModule(module_sp)) {
    if (m_dynamic_type)
      return m_dynamic_type.GetDisplayTypeName();
    return m_static_type.GetDisplayTypeName();
  }
  return ConstString();
}

TypeImpl TypeImpl::GetPointerType() const {
  ModuleSP module_sp;
  if (CheckModule(module_sp)) {
    if (m_dynamic_type.IsValid()) {
      return TypeImpl(m_static_type.GetPointerType(),
                      m_dynamic_type.GetPointerType());
    }
    return TypeImpl(m_static_type.GetPointerType());
  }
  return TypeImpl();
}

TypeImpl TypeImpl::GetPointeeType() const {
  ModuleSP module_sp;
  if (CheckModule(module_sp)) {
    if (m_dynamic_type.IsValid()) {
      return TypeImpl(m_static_type.GetPointeeType(),
                      m_dynamic_type.GetPointeeType());
    }
    return TypeImpl(m_static_type.GetPointeeType());
  }
  return TypeImpl();
}

TypeImpl TypeImpl::GetReferenceType() const {
  ModuleSP module_sp;
  if (CheckModule(module_sp)) {
    if (m_dynamic_type.IsValid()) {
      return TypeImpl(m_static_type.GetLValueReferenceType(),
                      m_dynamic_type.GetLValueReferenceType());
    }
    return TypeImpl(m_static_type.GetLValueReferenceType());
  }
  return TypeImpl();
}

TypeImpl TypeImpl::GetTypedefedType() const {
  ModuleSP module_sp;
  if (CheckModule(module_sp)) {
    if (m_dynamic_type.IsValid()) {
      return TypeImpl(m_static_type.GetTypedefedType(),
                      m_dynamic_type.GetTypedefedType());
    }
    return TypeImpl(m_static_type.GetTypedefedType());
  }
  return TypeImpl();
}

TypeImpl TypeImpl::GetDereferencedType() const {
  ModuleSP module_sp;
  if (CheckModule(module_sp)) {
    if (m_dynamic_type.IsValid()) {
      return TypeImpl(m_static_type.GetNonReferenceType(),
                      m_dynamic_type.GetNonReferenceType());
    }
    return TypeImpl(m_static_type.GetNonReferenceType());
  }
  return TypeImpl();
}

TypeImpl TypeImpl::GetUnqualifiedType() const {
  ModuleSP module_sp;
  if (CheckModule(module_sp)) {
    if (m_dynamic_type.IsValid()) {
      return TypeImpl(m_static_type.GetFullyUnqualifiedType(),
                      m_dynamic_type.GetFullyUnqualifiedType());
    }
    return TypeImpl(m_static_type.GetFullyUnqualifiedType());
  }
  return TypeImpl();
}

TypeImpl TypeImpl::GetCanonicalType() const {
  ModuleSP module_sp;
  if (CheckModule(module_sp)) {
    if (m_dynamic_type.IsValid()) {
      return TypeImpl(m_static_type.GetCanonicalType(),
                      m_dynamic_type.GetCanonicalType());
    }
    return TypeImpl(m_static_type.GetCanonicalType());
  }
  return TypeImpl();
}

CompilerType TypeImpl::GetCompilerType(bool prefer_dynamic) {
  ModuleSP module_sp;
  if (CheckModule(module_sp)) {
    if (prefer_dynamic) {
      if (m_dynamic_type.IsValid())
        return m_dynamic_type;
    }
    return m_static_type;
  }
  return CompilerType();
}

TypeSystem *TypeImpl::GetTypeSystem(bool prefer_dynamic) {
  ModuleSP module_sp;
  if (CheckModule(module_sp)) {
    if (prefer_dynamic) {
      if (m_dynamic_type.IsValid())
        return m_dynamic_type.GetTypeSystem();
    }
    return m_static_type.GetTypeSystem();
  }
  return nullptr;
}

bool TypeImpl::GetDescription(lldb_private::Stream &strm,
                              lldb::DescriptionLevel description_level) {
  ModuleSP module_sp;
  if (CheckModule(module_sp)) {
    if (m_dynamic_type.IsValid()) {
      strm.Printf("Dynamic:\n");
      m_dynamic_type.DumpTypeDescription(&strm);
      strm.Printf("\nStatic:\n");
    }
    m_static_type.DumpTypeDescription(&strm);
  } else {
    strm.PutCString("Invalid TypeImpl module for type has been deleted\n");
  }
  return true;
}

bool TypeMemberFunctionImpl::IsValid() {
  return m_type.IsValid() && m_kind != lldb::eMemberFunctionKindUnknown;
}

ConstString TypeMemberFunctionImpl::GetName() const { return m_name; }

ConstString TypeMemberFunctionImpl::GetMangledName() const {
  return m_decl.GetMangledName();
}

CompilerType TypeMemberFunctionImpl::GetType() const { return m_type; }

lldb::MemberFunctionKind TypeMemberFunctionImpl::GetKind() const {
  return m_kind;
}

bool TypeMemberFunctionImpl::GetDescription(Stream &stream) {
  switch (m_kind) {
  case lldb::eMemberFunctionKindUnknown:
    return false;
  case lldb::eMemberFunctionKindConstructor:
    stream.Printf("constructor for %s",
                  m_type.GetTypeName().AsCString("<unknown>"));
    break;
  case lldb::eMemberFunctionKindDestructor:
    stream.Printf("destructor for %s",
                  m_type.GetTypeName().AsCString("<unknown>"));
    break;
  case lldb::eMemberFunctionKindInstanceMethod:
    stream.Printf("instance method %s of type %s", m_name.AsCString(),
                  m_decl.GetDeclContext().GetName().AsCString());
    break;
  case lldb::eMemberFunctionKindStaticMethod:
    stream.Printf("static method %s of type %s", m_name.AsCString(),
                  m_decl.GetDeclContext().GetName().AsCString());
    break;
  }
  return true;
}

CompilerType TypeMemberFunctionImpl::GetReturnType() const {
  if (m_type)
    return m_type.GetFunctionReturnType();
  return m_decl.GetFunctionReturnType();
}

size_t TypeMemberFunctionImpl::GetNumArguments() const {
  if (m_type)
    return m_type.GetNumberOfFunctionArguments();
  else
    return m_decl.GetNumFunctionArguments();
}

CompilerType TypeMemberFunctionImpl::GetArgumentAtIndex(size_t idx) const {
  if (m_type)
    return m_type.GetFunctionArgumentAtIndex(idx);
  else
    return m_decl.GetFunctionArgumentType(idx);
}

TypeEnumMemberImpl::TypeEnumMemberImpl(const lldb::TypeImplSP &integer_type_sp,
                                       ConstString name,
                                       const llvm::APSInt &value)
    : m_integer_type_sp(integer_type_sp), m_name(name), m_value(value),
      m_valid((bool)name && (bool)integer_type_sp)

{}

//===-- PdbSymUid.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// A unique identification scheme for Pdb records.
// The scheme is to partition a 64-bit integer into an 8-bit tag field, which
// will contain some value from the PDB_SymType enumeration.  The format of the
// other 48-bits depend on the tag, but must be sufficient to locate the
// corresponding entry in the underlying PDB file quickly.  For example, for
// a compile unit, we use 2 bytes to represent the index, which allows fast
// access to the compile unit's information.
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SYMBOLFILENATIVEPDB_PDBSYMUID_H
#define LLDB_PLUGINS_SYMBOLFILENATIVEPDB_PDBSYMUID_H

#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/Support/Compiler.h"

#include "lldb/Utility/LLDBAssert.h"
#include "lldb/lldb-types.h"

namespace lldb_private {
namespace npdb {

// **important** - All concrete id types must have the 1-byte tag field at
// the beginning so that the types are all layout-compatible with each
// other, which is necessary in order to be able to safely access the tag
// member through any union member.

struct PdbCompilandId {
  uint64_t tag : 8;   // PDB_SymType::Compiland
  uint64_t modi : 16; // 0-based index of module in PDB
  uint64_t unused : 32;
};
struct PdbCuSymId {
  uint64_t tag : 8; // PDB_SymType::Data, Function, Block, etc.
  uint64_t
      offset : 32;    // Offset of symbol's record in module stream.  This is
                      // offset by 4 from the CVSymbolArray's notion of offset
                      // due to the debug magic at the beginning of the stream.
  uint64_t modi : 16; // 0-based index of module in PDB
};
struct PdbTypeSymId {
  uint64_t tag : 8;    // PDB_SymType::FunctionSig, Enum, PointerType, etc.
  uint64_t is_ipi : 8; // 1 if this value is from the IPI stream, 0 for TPI.
  uint64_t unused : 16;
  uint64_t index : 32; // codeview::TypeIndex
};

static_assert(sizeof(PdbCompilandId) == 8, "invalid uid size");
static_assert(sizeof(PdbCuSymId) == 8, "invalid uid size");
static_assert(std::is_standard_layout<PdbCompilandId>::value,
              "type is not standard layout!");
static_assert(std::is_standard_layout<PdbCuSymId>::value,
              "type is not standard layout!");

class PdbSymUid {
  union {
    PdbCompilandId comp_id;
    PdbCuSymId cu_sym;
    PdbTypeSymId type_sym;
  } m_uid;

  PdbSymUid() { ::memset(&m_uid, 0, sizeof(m_uid)); }

public:
  static bool isTypeSym(llvm::pdb::PDB_SymType tag) {
    switch (tag) {
    case llvm::pdb::PDB_SymType::ArrayType:
    case llvm::pdb::PDB_SymType::BaseClass:
    case llvm::pdb::PDB_SymType::BaseInterface:
    case llvm::pdb::PDB_SymType::BuiltinType:
    case llvm::pdb::PDB_SymType::CustomType:
    case llvm::pdb::PDB_SymType::Enum:
    case llvm::pdb::PDB_SymType::FunctionArg:
    case llvm::pdb::PDB_SymType::FunctionSig:
    case llvm::pdb::PDB_SymType::Typedef:
    case llvm::pdb::PDB_SymType::VectorType:
    case llvm::pdb::PDB_SymType::VTableShape:
    case llvm::pdb::PDB_SymType::PointerType:
    case llvm::pdb::PDB_SymType::UDT:
      return true;
    default:
      return false;
    }
  }

  static bool isCuSym(llvm::pdb::PDB_SymType tag) {
    switch (tag) {
    case llvm::pdb::PDB_SymType::Block:
    case llvm::pdb::PDB_SymType::Callee:
    case llvm::pdb::PDB_SymType::Caller:
    case llvm::pdb::PDB_SymType::CallSite:
    case llvm::pdb::PDB_SymType::CoffGroup:
    case llvm::pdb::PDB_SymType::CompilandDetails:
    case llvm::pdb::PDB_SymType::CompilandEnv:
    case llvm::pdb::PDB_SymType::Custom:
    case llvm::pdb::PDB_SymType::Data:
    case llvm::pdb::PDB_SymType::Function:
    case llvm::pdb::PDB_SymType::Inlinee:
    case llvm::pdb::PDB_SymType::InlineSite:
    case llvm::pdb::PDB_SymType::Label:
    case llvm::pdb::PDB_SymType::Thunk:
      return true;
    default:
      return false;
    }
  }

  static PdbSymUid makeCuSymId(llvm::codeview::ProcRefSym sym) {
    return makeCuSymId(llvm::pdb::PDB_SymType::Function, sym.Module - 1,
                       sym.SymOffset);
  }

  static PdbSymUid makeCuSymId(llvm::pdb::PDB_SymType type, uint16_t modi,
                               uint32_t offset) {
    lldbassert(isCuSym(type));

    PdbSymUid uid;
    uid.m_uid.cu_sym.modi = modi;
    uid.m_uid.cu_sym.offset = offset;
    uid.m_uid.cu_sym.tag = static_cast<uint8_t>(type);
    return uid;
  }

  static PdbSymUid makeCompilandId(llvm::codeview::ProcRefSym sym) {
    // S_PROCREF symbols are 1-based
    lldbassert(sym.Module > 0);
    return makeCompilandId(sym.Module - 1);
  }

  static PdbSymUid makeCompilandId(uint16_t modi) {
    PdbSymUid uid;
    uid.m_uid.comp_id.modi = modi;
    uid.m_uid.cu_sym.tag =
        static_cast<uint8_t>(llvm::pdb::PDB_SymType::Compiland);
    return uid;
  }

  static PdbSymUid makeTypeSymId(llvm::pdb::PDB_SymType type,
                                 llvm::codeview::TypeIndex index, bool is_ipi) {
    lldbassert(isTypeSym(type));

    PdbSymUid uid;
    uid.m_uid.type_sym.tag = static_cast<uint8_t>(type);
    uid.m_uid.type_sym.index = index.getIndex();
    uid.m_uid.type_sym.is_ipi = static_cast<uint8_t>(is_ipi);
    return uid;
  }

  static PdbSymUid fromOpaqueId(uint64_t value) {
    PdbSymUid result;
    ::memcpy(&result.m_uid, &value, sizeof(value));
    return result;
  }

  uint64_t toOpaqueId() const {
    uint64_t result;
    ::memcpy(&result, &m_uid, sizeof(m_uid));
    return result;
  }

  bool isPubSym() const {
    return tag() == llvm::pdb::PDB_SymType::PublicSymbol;
  }
  bool isCompiland() const {
    return tag() == llvm::pdb::PDB_SymType::Compiland;
  }

  llvm::pdb::PDB_SymType tag() const {
    return static_cast<llvm::pdb::PDB_SymType>(m_uid.comp_id.tag);
  }

  const PdbCompilandId &asCompiland() const {
    lldbassert(tag() == llvm::pdb::PDB_SymType::Compiland);
    return m_uid.comp_id;
  }

  const PdbCuSymId &asCuSym() const {
    lldbassert(isCuSym(tag()));
    return m_uid.cu_sym;
  }

  const PdbTypeSymId &asTypeSym() const {
    lldbassert(isTypeSym(tag()));
    return m_uid.type_sym;
  }
};

struct SymbolAndUid {
  llvm::codeview::CVSymbol sym;
  PdbSymUid uid;
};
} // namespace npdb
} // namespace lldb_private

#endif

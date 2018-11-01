//===-- PdbUtil.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PdbUtil.h"

#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"

#include "lldb/Utility/LLDBAssert.h"

#include "lldb/lldb-enumerations.h"

using namespace lldb_private;
using namespace lldb_private::npdb;
using namespace llvm::codeview;
using namespace llvm::pdb;

PDB_SymType lldb_private::npdb::CVSymToPDBSym(SymbolKind kind) {
  switch (kind) {
  case S_COMPILE3:
  case S_OBJNAME:
    return PDB_SymType::CompilandDetails;
  case S_ENVBLOCK:
    return PDB_SymType::CompilandEnv;
  case S_THUNK32:
  case S_TRAMPOLINE:
    return PDB_SymType::Thunk;
  case S_COFFGROUP:
    return PDB_SymType::CoffGroup;
  case S_EXPORT:
    return PDB_SymType::Export;
  case S_LPROC32:
  case S_GPROC32:
  case S_LPROC32_DPC:
    return PDB_SymType::Function;
  case S_PUB32:
    return PDB_SymType::PublicSymbol;
  case S_INLINESITE:
    return PDB_SymType::InlineSite;
  case S_LOCAL:
  case S_BPREL32:
  case S_REGREL32:
  case S_MANCONSTANT:
  case S_CONSTANT:
  case S_LDATA32:
  case S_GDATA32:
  case S_LMANDATA:
  case S_GMANDATA:
  case S_LTHREAD32:
  case S_GTHREAD32:
    return PDB_SymType::Data;
  case S_BLOCK32:
    return PDB_SymType::Block;
  case S_LABEL32:
    return PDB_SymType::Label;
  case S_CALLSITEINFO:
    return PDB_SymType::CallSite;
  case S_HEAPALLOCSITE:
    return PDB_SymType::HeapAllocationSite;
  case S_CALLEES:
    return PDB_SymType::Callee;
  case S_CALLERS:
    return PDB_SymType::Caller;
  default:
    lldbassert(false && "Invalid symbol record kind!");
  }
  return PDB_SymType::None;
}

PDB_SymType lldb_private::npdb::CVTypeToPDBType(TypeLeafKind kind) {
  switch (kind) {
  case LF_ARRAY:
    return PDB_SymType::ArrayType;
  case LF_ARGLIST:
    return PDB_SymType::FunctionSig;
  case LF_BCLASS:
    return PDB_SymType::BaseClass;
  case LF_BINTERFACE:
    return PDB_SymType::BaseInterface;
  case LF_CLASS:
  case LF_STRUCTURE:
  case LF_INTERFACE:
  case LF_UNION:
    return PDB_SymType::UDT;
  case LF_POINTER:
    return PDB_SymType::PointerType;
  case LF_ENUM:
    return PDB_SymType::Enum;
  case LF_PROCEDURE:
    return PDB_SymType::FunctionSig;
  default:
    lldbassert(false && "Invalid type record kind!");
  }
  return PDB_SymType::None;
}

bool lldb_private::npdb::SymbolHasAddress(const CVSymbol &sym) {
  switch (sym.kind()) {
  case S_GPROC32:
  case S_LPROC32:
  case S_GPROC32_ID:
  case S_LPROC32_ID:
  case S_LPROC32_DPC:
  case S_LPROC32_DPC_ID:
  case S_THUNK32:
  case S_TRAMPOLINE:
  case S_COFFGROUP:
  case S_BLOCK32:
  case S_LABEL32:
  case S_CALLSITEINFO:
  case S_HEAPALLOCSITE:
  case S_LDATA32:
  case S_GDATA32:
  case S_LMANDATA:
  case S_GMANDATA:
  case S_LTHREAD32:
  case S_GTHREAD32:
    return true;
  default:
    return false;
  }
}

bool lldb_private::npdb::SymbolIsCode(const CVSymbol &sym) {
  switch (sym.kind()) {
  case S_GPROC32:
  case S_LPROC32:
  case S_GPROC32_ID:
  case S_LPROC32_ID:
  case S_LPROC32_DPC:
  case S_LPROC32_DPC_ID:
  case S_THUNK32:
  case S_TRAMPOLINE:
  case S_COFFGROUP:
  case S_BLOCK32:
    return true;
  default:
    return false;
  }
}

template <typename RecordT> RecordT createRecord(const CVSymbol &sym) {
  RecordT record(static_cast<SymbolRecordKind>(sym.kind()));
  cantFail(SymbolDeserializer::deserializeAs<RecordT>(sym, record));
  return record;
}

template <typename RecordT>
static SegmentOffset GetSegmentAndOffset(const CVSymbol &sym) {
  RecordT record = createRecord<RecordT>(sym);
  return {record.Segment, record.CodeOffset};
}

template <>
SegmentOffset GetSegmentAndOffset<TrampolineSym>(const CVSymbol &sym) {
  TrampolineSym record = createRecord<TrampolineSym>(sym);
  return {record.ThunkSection, record.ThunkOffset};
}

template <> SegmentOffset GetSegmentAndOffset<Thunk32Sym>(const CVSymbol &sym) {
  Thunk32Sym record = createRecord<Thunk32Sym>(sym);
  return {record.Segment, record.Offset};
}

template <>
SegmentOffset GetSegmentAndOffset<CoffGroupSym>(const CVSymbol &sym) {
  CoffGroupSym record = createRecord<CoffGroupSym>(sym);
  return {record.Segment, record.Offset};
}

template <> SegmentOffset GetSegmentAndOffset<DataSym>(const CVSymbol &sym) {
  DataSym record = createRecord<DataSym>(sym);
  return {record.Segment, record.DataOffset};
}

template <>
SegmentOffset GetSegmentAndOffset<ThreadLocalDataSym>(const CVSymbol &sym) {
  ThreadLocalDataSym record = createRecord<ThreadLocalDataSym>(sym);
  return {record.Segment, record.DataOffset};
}

SegmentOffset lldb_private::npdb::GetSegmentAndOffset(const CVSymbol &sym) {
  switch (sym.kind()) {
  case S_GPROC32:
  case S_LPROC32:
  case S_GPROC32_ID:
  case S_LPROC32_ID:
  case S_LPROC32_DPC:
  case S_LPROC32_DPC_ID:
    return ::GetSegmentAndOffset<ProcSym>(sym);
  case S_THUNK32:
    return ::GetSegmentAndOffset<Thunk32Sym>(sym);
    break;
  case S_TRAMPOLINE:
    return ::GetSegmentAndOffset<TrampolineSym>(sym);
    break;
  case S_COFFGROUP:
    return ::GetSegmentAndOffset<CoffGroupSym>(sym);
    break;
  case S_BLOCK32:
    return ::GetSegmentAndOffset<BlockSym>(sym);
    break;
  case S_LABEL32:
    return ::GetSegmentAndOffset<LabelSym>(sym);
    break;
  case S_CALLSITEINFO:
    return ::GetSegmentAndOffset<CallSiteInfoSym>(sym);
    break;
  case S_HEAPALLOCSITE:
    return ::GetSegmentAndOffset<HeapAllocationSiteSym>(sym);
    break;
  case S_LDATA32:
  case S_GDATA32:
  case S_LMANDATA:
  case S_GMANDATA:
    return ::GetSegmentAndOffset<DataSym>(sym);
    break;
  case S_LTHREAD32:
  case S_GTHREAD32:
    return ::GetSegmentAndOffset<ThreadLocalDataSym>(sym);
    break;
  default:
    lldbassert(false && "Record does not have a segment/offset!");
  }
  return {0, 0};
}

template <typename RecordT>
SegmentOffsetLength GetSegmentOffsetAndLength(const CVSymbol &sym) {
  RecordT record = createRecord<RecordT>(sym);
  return {record.Segment, record.CodeOffset, record.CodeSize};
}

template <>
SegmentOffsetLength
GetSegmentOffsetAndLength<TrampolineSym>(const CVSymbol &sym) {
  TrampolineSym record = createRecord<TrampolineSym>(sym);
  return {record.ThunkSection, record.ThunkOffset, record.Size};
}

template <>
SegmentOffsetLength GetSegmentOffsetAndLength<Thunk32Sym>(const CVSymbol &sym) {
  Thunk32Sym record = createRecord<Thunk32Sym>(sym);
  return SegmentOffsetLength{record.Segment, record.Offset, record.Length};
}

template <>
SegmentOffsetLength
GetSegmentOffsetAndLength<CoffGroupSym>(const CVSymbol &sym) {
  CoffGroupSym record = createRecord<CoffGroupSym>(sym);
  return SegmentOffsetLength{record.Segment, record.Offset, record.Size};
}

SegmentOffsetLength
lldb_private::npdb::GetSegmentOffsetAndLength(const CVSymbol &sym) {
  switch (sym.kind()) {
  case S_GPROC32:
  case S_LPROC32:
  case S_GPROC32_ID:
  case S_LPROC32_ID:
  case S_LPROC32_DPC:
  case S_LPROC32_DPC_ID:
    return ::GetSegmentOffsetAndLength<ProcSym>(sym);
  case S_THUNK32:
    return ::GetSegmentOffsetAndLength<Thunk32Sym>(sym);
    break;
  case S_TRAMPOLINE:
    return ::GetSegmentOffsetAndLength<TrampolineSym>(sym);
    break;
  case S_COFFGROUP:
    return ::GetSegmentOffsetAndLength<CoffGroupSym>(sym);
    break;
  case S_BLOCK32:
    return ::GetSegmentOffsetAndLength<BlockSym>(sym);
    break;
  default:
    lldbassert(false && "Record does not have a segment/offset/length triple!");
  }
  return {0, 0, 0};
}

bool lldb_private::npdb::IsForwardRefUdt(CVType cvt) {
  ClassRecord cr;
  UnionRecord ur;
  EnumRecord er;
  switch (cvt.kind()) {
  case LF_CLASS:
  case LF_STRUCTURE:
  case LF_INTERFACE:
    llvm::cantFail(TypeDeserializer::deserializeAs<ClassRecord>(cvt, cr));
    return cr.isForwardRef();
  case LF_UNION:
    llvm::cantFail(TypeDeserializer::deserializeAs<UnionRecord>(cvt, ur));
    return ur.isForwardRef();
  case LF_ENUM:
    llvm::cantFail(TypeDeserializer::deserializeAs<EnumRecord>(cvt, er));
    return er.isForwardRef();
  default:
    return false;
  }
}

lldb::AccessType
lldb_private::npdb::TranslateMemberAccess(MemberAccess access) {
  switch (access) {
  case MemberAccess::Private:
    return lldb::eAccessPrivate;
  case MemberAccess::Protected:
    return lldb::eAccessProtected;
  case MemberAccess::Public:
    return lldb::eAccessPublic;
  case MemberAccess::None:
    return lldb::eAccessNone;
  }
  llvm_unreachable("unreachable");
}

TypeIndex lldb_private::npdb::GetFieldListIndex(CVType cvt) {
  switch (cvt.kind()) {
  case LF_CLASS:
  case LF_STRUCTURE:
  case LF_INTERFACE: {
    ClassRecord cr;
    cantFail(TypeDeserializer::deserializeAs<ClassRecord>(cvt, cr));
    return cr.FieldList;
  }
  case LF_UNION: {
    UnionRecord ur;
    cantFail(TypeDeserializer::deserializeAs<UnionRecord>(cvt, ur));
    return ur.FieldList;
  }
  case LF_ENUM: {
    EnumRecord er;
    cantFail(TypeDeserializer::deserializeAs<EnumRecord>(cvt, er));
    return er.FieldList;
  }
  default:
    llvm_unreachable("Unreachable!");
  }
}

TypeIndex lldb_private::npdb::LookThroughModifierRecord(CVType modifier) {
  lldbassert(modifier.kind() == LF_MODIFIER);
  ModifierRecord mr;
  llvm::cantFail(TypeDeserializer::deserializeAs<ModifierRecord>(modifier, mr));
  return mr.ModifiedType;
}

llvm::StringRef lldb_private::npdb::DropNameScope(llvm::StringRef name) {
  // Not all PDB names can be parsed with CPlusPlusNameParser.
  // E.g. it fails on names containing `anonymous namespace'.
  // So we simply drop everything before '::'

  auto offset = name.rfind("::");
  if (offset == llvm::StringRef::npos)
    return name;
  assert(offset + 2 <= name.size());

  return name.substr(offset + 2);
}

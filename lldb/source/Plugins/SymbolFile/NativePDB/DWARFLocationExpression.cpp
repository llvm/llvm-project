//===-- DWARFLocationExpression.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFLocationExpression.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamBuffer.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/Support/Endian.h"

#include "PdbUtil.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::npdb;
using namespace llvm::codeview;
using namespace llvm::pdb;

static bool IsSimpleTypeSignedInteger(SimpleTypeKind kind) {
  switch (kind) {
  case SimpleTypeKind::Int128:
  case SimpleTypeKind::Int64:
  case SimpleTypeKind::Int64Quad:
  case SimpleTypeKind::Int32:
  case SimpleTypeKind::Int32Long:
  case SimpleTypeKind::Int16:
  case SimpleTypeKind::Int16Short:
  case SimpleTypeKind::Float128:
  case SimpleTypeKind::Float80:
  case SimpleTypeKind::Float64:
  case SimpleTypeKind::Float32:
  case SimpleTypeKind::Float16:
  case SimpleTypeKind::NarrowCharacter:
  case SimpleTypeKind::SignedCharacter:
  case SimpleTypeKind::SByte:
    return true;
  default:
    return false;
  }
}

static std::pair<size_t, bool> GetIntegralTypeInfo(TypeIndex ti,
                                                   TpiStream &tpi) {
  if (ti.isSimple()) {
    SimpleTypeKind stk = ti.getSimpleKind();
    return {GetTypeSizeForSimpleKind(stk), IsSimpleTypeSignedInteger(stk)};
  }

  CVType cvt = tpi.getType(ti);
  switch (cvt.kind()) {
  case LF_MODIFIER: {
    ModifierRecord mfr;
    llvm::cantFail(TypeDeserializer::deserializeAs<ModifierRecord>(cvt, mfr));
    return GetIntegralTypeInfo(mfr.ModifiedType, tpi);
  }
  case LF_POINTER: {
    PointerRecord pr;
    llvm::cantFail(TypeDeserializer::deserializeAs<PointerRecord>(cvt, pr));
    return GetIntegralTypeInfo(pr.ReferentType, tpi);
  }
  case LF_ENUM: {
    EnumRecord er;
    llvm::cantFail(TypeDeserializer::deserializeAs<EnumRecord>(cvt, er));
    return GetIntegralTypeInfo(er.UnderlyingType, tpi);
  }
  default:
    assert(false && "Type is not integral!");
    return {0, false};
  }
}

template <typename StreamWriter>
static DWARFExpression MakeLocationExpressionInternal(lldb::ModuleSP module,
                                                      StreamWriter &&writer) {
  const ArchSpec &architecture = module->GetArchitecture();
  ByteOrder byte_order = architecture.GetByteOrder();
  uint32_t address_size = architecture.GetAddressByteSize();
  uint32_t byte_size = architecture.GetDataByteSize();
  if (byte_order == eByteOrderInvalid || address_size == 0)
    return DWARFExpression(nullptr);

  RegisterKind register_kind = eRegisterKindDWARF;
  StreamBuffer<32> stream(Stream::eBinary, address_size, byte_order);

  if (!writer(stream, register_kind))
    return DWARFExpression(nullptr);

  DataBufferSP buffer =
      std::make_shared<DataBufferHeap>(stream.GetData(), stream.GetSize());
  DataExtractor extractor(buffer, byte_order, address_size, byte_size);
  DWARFExpression result(module, extractor, nullptr, 0, buffer->GetByteSize());
  result.SetRegisterKind(register_kind);

  return result;
}

DWARFExpression lldb_private::npdb::MakeGlobalLocationExpression(
    uint16_t section, uint32_t offset, ModuleSP module) {
  assert(section > 0);
  assert(module);

  return MakeLocationExpressionInternal(
      module, [&](Stream &stream, RegisterKind &register_kind) -> bool {
        stream.PutHex8(llvm::dwarf::DW_OP_addr);

        SectionList *section_list = module->GetSectionList();
        assert(section_list);

        // Section indices in PDB are 1-based, but in DWARF they are 0-based, so
        // we need to subtract 1.
        uint32_t section_idx = section - 1;
        if (section_idx >= section_list->GetSize())
          return false;

        auto section_ptr = section_list->GetSectionAtIndex(section_idx);
        if (!section_ptr)
          return false;

        stream.PutMaxHex64(section_ptr->GetFileAddress() + offset,
                           stream.GetAddressByteSize(), stream.GetByteOrder());

        return true;
      });
}

DWARFExpression lldb_private::npdb::MakeConstantLocationExpression(
    TypeIndex underlying_ti, TpiStream &tpi, const llvm::APSInt &constant,
    ModuleSP module) {
  const ArchSpec &architecture = module->GetArchitecture();
  uint32_t address_size = architecture.GetAddressByteSize();

  size_t size = 0;
  bool is_signed = false;
  std::tie(size, is_signed) = GetIntegralTypeInfo(underlying_ti, tpi);

  union {
    llvm::support::little64_t I;
    llvm::support::ulittle64_t U;
  } Value;

  std::shared_ptr<DataBufferHeap> buffer = std::make_shared<DataBufferHeap>();
  buffer->SetByteSize(size);

  llvm::ArrayRef<uint8_t> bytes;
  if (is_signed) {
    Value.I = constant.getSExtValue();
  } else {
    Value.U = constant.getZExtValue();
  }

  bytes = llvm::makeArrayRef(reinterpret_cast<const uint8_t *>(&Value), 8)
              .take_front(size);
  buffer->CopyData(bytes.data(), size);
  DataExtractor extractor(buffer, lldb::eByteOrderLittle, address_size);
  DWARFExpression result(nullptr, extractor, nullptr, 0, size);
  return result;
}

//===- Dwarf2BTF.cpp ------------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DwarfUnit.h"
#include "Dwarf2BTF.h"
#include "llvm/MC/MCBTFContext.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"

namespace llvm {

unsigned char Die2BTFEntry::getDieKind(const DIE & Die) {
  auto Tag = Die.getTag();

  switch (Tag) {
    case dwarf::DW_TAG_base_type:
      if (getBaseTypeEncoding(Die) == BTF_INVALID_ENCODING)
        return BTF_KIND_UNKN;
      return BTF_KIND_INT;
    case dwarf::DW_TAG_const_type:
      return BTF_KIND_CONST;
    case dwarf::DW_TAG_pointer_type:
      return BTF_KIND_PTR;
    case dwarf::DW_TAG_restrict_type:
      return BTF_KIND_RESTRICT;
    case dwarf::DW_TAG_volatile_type:
      return BTF_KIND_VOLATILE;
    case dwarf::DW_TAG_typedef:
      return BTF_KIND_TYPEDEF;
    case dwarf::DW_TAG_structure_type:
    case dwarf::DW_TAG_class_type:
      if (Die.findAttribute(dwarf::DW_AT_declaration).getType()
          != DIEValue::isNone)
        return BTF_KIND_FWD;
      else
        return BTF_KIND_STRUCT;
    case dwarf::DW_TAG_union_type:
      if (Die.findAttribute(dwarf::DW_AT_declaration).getType()
          != DIEValue::isNone)
        return BTF_KIND_FWD;
      else
        return BTF_KIND_UNION;
    case dwarf::DW_TAG_enumeration_type:
      return BTF_KIND_ENUM;
    case dwarf::DW_TAG_array_type:
      return BTF_KIND_ARRAY;
    case dwarf::DW_TAG_subprogram:
      return BTF_KIND_FUNC;
    case dwarf::DW_TAG_subroutine_type:
      return BTF_KIND_FUNC_PROTO;
    default:
      break;
  }

  return BTF_KIND_UNKN;
}

std::unique_ptr<Die2BTFEntry> Die2BTFEntry::dieToBTFTypeEntry(const DIE &Die) {
  unsigned char Kind = getDieKind(Die);

  switch (Kind) {
    case BTF_KIND_INT:
      return make_unique<Die2BTFEntryInt>(Die);
    case BTF_KIND_PTR:
    case BTF_KIND_TYPEDEF:
    case BTF_KIND_VOLATILE:
    case BTF_KIND_CONST:
    case BTF_KIND_RESTRICT:
    case BTF_KIND_FWD:
      return make_unique<Die2BTFEntry>(Die);
    case BTF_KIND_ARRAY:
      return make_unique<Die2BTFEntryArray>(Die);
    case BTF_KIND_STRUCT:
    case BTF_KIND_UNION:
      return make_unique<Die2BTFEntryStruct>(Die);
    case BTF_KIND_ENUM:
      return make_unique<Die2BTFEntryEnum>(Die);
    case BTF_KIND_FUNC:
    case BTF_KIND_FUNC_PROTO:
      return make_unique<Die2BTFEntryFunc>(Die);
    default:
      break;
  }
  return nullptr;
}

bool Die2BTFEntry::shouldSkipDie(const DIE &Die) {
  auto Tag = Die.getTag();

  switch (Tag) {
    case dwarf::DW_TAG_const_type:
    case dwarf::DW_TAG_pointer_type:
    case dwarf::DW_TAG_restrict_type:
    case dwarf::DW_TAG_typedef:
    case dwarf::DW_TAG_volatile_type:
    {
      auto TypeV = Die.findAttribute(dwarf::DW_AT_type);
      if (TypeV.getType() == DIEValue::isNone)
        return false;
      auto &TypeDie = TypeV.getDIEEntry().getEntry();
      return Die2BTFEntry::shouldSkipDie(TypeDie);
    }
    default:
      return getDieKind(Die) == BTF_KIND_UNKN;
  }
  return true;
}
unsigned char Die2BTFEntry::getBaseTypeEncoding(const DIE &Die) {
  auto V = Die.findAttribute(dwarf::DW_AT_encoding);

  if (V.getType() != DIEValue::isInteger)
    return BTF_INVALID_ENCODING;

  switch (V.getDIEInteger().getValue()) {
    case dwarf::DW_ATE_boolean:
      return BTF_INT_BOOL;
    case dwarf::DW_ATE_signed:
      return BTF_INT_SIGNED;
    case dwarf::DW_ATE_signed_char:
      return BTF_INT_CHAR;
    case dwarf::DW_ATE_unsigned:
      return 0;
    case dwarf::DW_ATE_unsigned_char:
      return BTF_INT_CHAR;
    case dwarf::DW_ATE_imaginary_float:
    case dwarf::DW_ATE_packed_decimal:
    case dwarf::DW_ATE_numeric_string:
    case dwarf::DW_ATE_edited:
    case dwarf::DW_ATE_signed_fixed:
    case dwarf::DW_ATE_address:
    case dwarf::DW_ATE_complex_float:
    case dwarf::DW_ATE_float:
    default:
      break;
  }
  return BTF_INVALID_ENCODING;
}

Die2BTFEntry::Die2BTFEntry(const DIE &Die) : Die(Die) {
  unsigned char Kind = getDieKind(Die);

  switch (Kind) {
    case BTF_KIND_CONST:
    case BTF_KIND_FWD:
    case BTF_KIND_PTR:
    case BTF_KIND_RESTRICT:
    case BTF_KIND_TYPEDEF:
    case BTF_KIND_VOLATILE:
      break;
    default:
      assert("Invalid Die passed into BTFTypeEntry()");
      break;
  }

  BTFType.info = (Kind & 0xf) << 24;
}

void Die2BTFEntry::completeData(class Dwarf2BTF &Dwarf2BTF) {
    auto TypeV = Die.findAttribute(dwarf::DW_AT_type);
    if (TypeV.getType() == DIEValue::isNone) {
      BTFType.type = 0;
    } else {
      auto &TypeDie = TypeV.getDIEEntry().getEntry();
      auto Type = Dwarf2BTF.getTypeIndex(TypeDie);
      BTFType.type = Type;
    }

    unsigned char Kind = getDieKind(Die);
    if (Kind != BTF_KIND_FWD) {
      BTFType.name_off = 0;
    } else {
      auto NameV = Die.findAttribute(dwarf::DW_AT_name);
      auto Str = NameV.getDIEString().getString();
      BTFType.name_off = Dwarf2BTF.addBTFString(Str);
    }

    auto typeEntry = make_unique<BTFTypeEntry>(Id, BTFType);
    Dwarf2BTF.addBTFTypeEntry(std::move(typeEntry));
}

Die2BTFEntryInt::Die2BTFEntryInt(const DIE &Die) : Die2BTFEntry(Die) {
  unsigned char Kind = getDieKind(Die);

  switch (Kind) {
    case BTF_KIND_INT:
      break;
    default:
      assert("Invalid Die passed into BTFTypeEntryInt()");
      break;
  }

  // handle BTF_INT_ENCODING in IntVal
  auto Encoding = Die2BTFEntry::getBaseTypeEncoding(Die);
  assert((Encoding != BTF_INVALID_ENCODING) &&
         "Invalid Die passed to BTFTypeEntryInt()");
  __u32 IntVal = (Encoding & 0xf) << 24;

  // handle BTF_INT_OFFSET in IntVal
  auto V = Die.findAttribute(dwarf::DW_AT_bit_offset);
  if (V.getType() == DIEValue::isInteger)
    IntVal |= (V.getDIEInteger().getValue() & 0xff) << 16;

  // get btf_type.size
  V = Die.findAttribute(dwarf::DW_AT_byte_size);
  __u32 Size = V.getDIEInteger().getValue() & 0xffffffff;

// handle BTF_INT_BITS in IntVal
  V = Die.findAttribute(dwarf::DW_AT_bit_size);
  if (V.getType() == DIEValue::isInteger)
    IntVal |= V.getDIEInteger().getValue() & 0xff;
  else
    IntVal |= (Size << 3) & 0xff;

  BTFType.info = BTF_KIND_INT << 24;
  BTFType.size = Size;
  this->IntVal = IntVal;
}

void Die2BTFEntryInt::completeData(class Dwarf2BTF &Dwarf2BTF) {
    auto NameV = Die.findAttribute(dwarf::DW_AT_name);
    auto TypeV = Die.findAttribute(dwarf::DW_AT_type);
    auto Str = NameV.getDIEString().getString();

    BTFType.name_off = Dwarf2BTF.addBTFString(Str);

    auto typeEntry = make_unique<BTFTypeEntryInt>(Id, BTFType, IntVal);
    Dwarf2BTF.addBTFTypeEntry(std::move(typeEntry));
}

Die2BTFEntryEnum::Die2BTFEntryEnum(const DIE &Die) : Die2BTFEntry(Die) {
  // get btf_type.size
  auto V = Die.findAttribute(dwarf::DW_AT_byte_size);
  __u32 Size = V.getDIEInteger().getValue() & 0xffffffff;

  int Vlen = 0;
  for (auto &ChildDie : Die.children())
    if (ChildDie.getTag() == dwarf::DW_TAG_enumerator)
      Vlen++;

  BTFType.info = (BTF_KIND_ENUM << 24) | (Vlen & BTF_MAX_VLEN);
  BTFType.type = Size;
}

void Die2BTFEntryEnum::completeData(class Dwarf2BTF &Dwarf2BTF) {
  auto TypeV = Die.findAttribute(dwarf::DW_AT_type);
  auto NameV = Die.findAttribute(dwarf::DW_AT_name);

  if (NameV.getType() != DIEValue::isNone) {
    auto Str = NameV.getDIEString().getString();
    BTFType.name_off = Dwarf2BTF.addBTFString(Str);
  } else
    BTFType.name_off = 0;

  for (auto &ChildDie : Die.children()) {
    struct btf_enum BTFEnum;
    auto ChildNameV = ChildDie.findAttribute(dwarf::DW_AT_name);
    auto Str = ChildNameV.getDIEString().getString();

    BTFEnum.name_off = Dwarf2BTF.addBTFString(Str);
    auto ChildValueV = ChildDie.findAttribute(dwarf::DW_AT_const_value);
    BTFEnum.val = (__s32)(ChildValueV.getDIEInteger().getValue());

    EnumValues.push_back(BTFEnum);
  }

  auto typeEntry = make_unique<BTFTypeEntryEnum>(Id, BTFType, EnumValues);
  Dwarf2BTF.addBTFTypeEntry(std::move(typeEntry));
}

Die2BTFEntryArray::Die2BTFEntryArray(const DIE &Die) :
    Die2BTFEntry(Die) {
  BTFType.info = (BTF_KIND_ARRAY << 24);
  BTFType.size = 0;
}

void Die2BTFEntryArray::completeData(class Dwarf2BTF &Dwarf2BTF) {
  auto NameV = Die.findAttribute(dwarf::DW_AT_name);

  std::string Str;
  if (NameV.getType() != DIEValue::isNone)
    Str = NameV.getDIEString().getString();
  BTFType.name_off = Dwarf2BTF.addBTFString(Str);

  auto &ArrayTypeDie = Die.findAttribute(dwarf::DW_AT_type).getDIEEntry().getEntry();
  ArrayInfo.type = Dwarf2BTF.getTypeIndex(ArrayTypeDie);

  // The number of elements should count all subranges
  unsigned Nelems = 1;
  bool IsFirstSubrange = true;
  for (auto &ChildDie : Die.children()) {
    if (ChildDie.getTag() == dwarf::DW_TAG_subrange_type) {
      if (IsFirstSubrange) {
        auto TypeV = ChildDie.findAttribute(dwarf::DW_AT_type);
        auto &TypeDie = TypeV.getDIEEntry().getEntry();
        ArrayInfo.index_type = Dwarf2BTF.getTypeIndex(TypeDie);
        IsFirstSubrange = false;
      }
      auto CountV = ChildDie.findAttribute(dwarf::DW_AT_count);
      if (CountV.getType() == DIEValue::isNone) {
        // array like a[] which essentially a pointer
        Nelems = 0;
        break;
      }
      Nelems *= (__u32)(CountV.getDIEInteger().getValue());
    }
  }
  ArrayInfo.nelems = Nelems;

  auto TypeEntry = make_unique<BTFTypeEntryArray>(Id, BTFType, ArrayInfo);
  Dwarf2BTF.addBTFTypeEntry(std::move(TypeEntry));
}

Die2BTFEntryStruct::Die2BTFEntryStruct(const DIE &Die) : Die2BTFEntry(Die) {
  // get btf_type.size
  auto V = Die.findAttribute(dwarf::DW_AT_byte_size);
  __u32 Size = V.getDIEInteger().getValue() & 0xffffffff;
  auto Kind = Die2BTFEntry::getDieKind(Die);

  int Vlen = 0;
  for (auto &ChildDie : Die.children())
    if (ChildDie.getTag() == dwarf::DW_TAG_member)
      Vlen++;

  BTFType.size = Size;
  BTFType.info = (Kind << 24) | (Vlen & BTF_MAX_VLEN);
}

void Die2BTFEntryStruct::completeData(class Dwarf2BTF &Dwarf2BTF) {
  auto NameV = Die.findAttribute(dwarf::DW_AT_name);

  if (NameV.getType() != DIEValue::isNone) {
    auto Str = NameV.getDIEString().getString();
    BTFType.name_off = Dwarf2BTF.addBTFString(Str);
  } else
    BTFType.name_off = 0;


  for (auto &ChildDie : Die.children()) {
    if (ChildDie.getTag() != dwarf::DW_TAG_member)
      continue;

    struct btf_member BTFMember;
    auto ChildNameV = ChildDie.findAttribute(dwarf::DW_AT_name);

    if (ChildNameV.getType() != DIEValue::isNone) {
      auto Str = ChildNameV.getDIEString().getString();
      BTFMember.name_off = Dwarf2BTF.addBTFString(Str);
    } else
      BTFMember.name_off = 0;

    auto TypeV = ChildDie.findAttribute(dwarf::DW_AT_type);
    auto &TypeDie = TypeV.getDIEEntry().getEntry();
    BTFMember.type = Dwarf2BTF.getTypeIndex(TypeDie);

    auto MemLocV = ChildDie.findAttribute(dwarf::DW_AT_data_member_location);
    unsigned MemLoc = MemLocV.getDIEInteger().getValue() * 8;

    auto ByteSizeV = ChildDie.findAttribute(dwarf::DW_AT_byte_size);
    if (ByteSizeV.getType() != DIEValue::isNone) {
      unsigned ByteSize = ByteSizeV.getDIEInteger().getValue();
      auto BitOffsetV = ChildDie.findAttribute(dwarf::DW_AT_bit_offset);
      unsigned BitOffset = BitOffsetV.getDIEInteger().getValue();
      auto BitSizeV = ChildDie.findAttribute(dwarf::DW_AT_bit_size);
      unsigned BitSize = BitSizeV.getDIEInteger().getValue();
      if (Dwarf2BTF.isLittleEndian())
        MemLoc += ByteSize * 8 - BitSize - BitOffset;
      else
        MemLoc += BitOffset;
    }
    BTFMember.offset = MemLoc;

    Members.push_back(BTFMember);
  }

  auto typeEntry = make_unique<BTFTypeEntryStruct>(Id, BTFType, Members);
  Dwarf2BTF.addBTFTypeEntry(std::move(typeEntry));
}

Die2BTFEntryFunc::Die2BTFEntryFunc(const DIE &Die) : Die2BTFEntry(Die) {
  auto Kind = Die2BTFEntry::getDieKind(Die);

  int Vlen = 0;
  for (auto &ChildDie : Die.children())
    if (ChildDie.getTag() == dwarf::DW_TAG_formal_parameter)
      Vlen++;

  BTFType.size = 0;
  BTFType.info = (Kind << 24) | (Vlen & BTF_MAX_VLEN);
}

void Die2BTFEntryFunc::completeData(class Dwarf2BTF &Dwarf2BTF) {
  auto NameV = Die.findAttribute(dwarf::DW_AT_name);
  if (NameV.getType() == DIEValue::isNone) {
    BTFType.name_off = 0;
  } else {
    auto Str = NameV.getDIEString().getString();
    BTFType.name_off = Dwarf2BTF.addBTFString(Str);
  }

  auto RetTypeV = Die.findAttribute(dwarf::DW_AT_type);
  if (RetTypeV.getType() != DIEValue::isNone) {
    auto &TypeDie = RetTypeV.getDIEEntry().getEntry();
    BTFType.type = Dwarf2BTF.getTypeIndex(TypeDie);
  } else {
    BTFType.type = 0;
  }

  for (auto &ChildDie : Die.children()) {
    if (ChildDie.getTag() == dwarf::DW_TAG_formal_parameter) {
      auto TypeV = ChildDie.findAttribute(dwarf::DW_AT_abstract_origin);
      if (TypeV.getType() != DIEValue::isNone) {
        auto &AbsOriginDie = TypeV.getDIEEntry().getEntry();
        assert(AbsOriginDie.getTag() == dwarf::DW_TAG_formal_parameter);
        TypeV = AbsOriginDie.findAttribute(dwarf::DW_AT_type);
      } else {
        TypeV = ChildDie.findAttribute(dwarf::DW_AT_type);
      }
      auto &TypeDie = TypeV.getDIEEntry().getEntry();
      Parameters.push_back(Dwarf2BTF.getTypeIndex(TypeDie));
    } else if (ChildDie.getTag() == dwarf::DW_TAG_unspecified_parameters) {
      Parameters.push_back(0);
    }
  }

  auto typeEntry = make_unique<BTFTypeEntryFunc>(Id, BTFType, Parameters);
  Dwarf2BTF.addBTFTypeEntry(std::move(typeEntry));

  if (BTF_INFO_KIND(BTFType.info) == BTF_KIND_FUNC) {
    auto LowPCV = Die.findAttribute(dwarf::DW_AT_low_pc);
    if (LowPCV.getType() != DIEValue::isNone) {
      const MCSymbol *Label = LowPCV.getDIELabel().getValue();
      BTFFuncInfo FuncInfo;
      unsigned SecNameOff;

      FuncInfo.Label = Label;
      FuncInfo.TypeId = Id;
      if (Label->isInSection()) {
        MCSection &Section = Label->getSection();
        MCSectionELF *SectionELF = dyn_cast<MCSectionELF>(&Section);
        assert(SectionELF);
        SecNameOff = Dwarf2BTF.addBTFString(SectionELF->getSectionName().str());
      } else {
        SecNameOff = Dwarf2BTF.addBTFString(".text");
      }
      Dwarf2BTF.addBTFFuncInfo(SecNameOff, FuncInfo);
    }
  }
}

Dwarf2BTF::Dwarf2BTF(MCContext &Context, bool IsLittleEndian)
  : OuterCtx(Context), IsLE(IsLittleEndian) {
  BTFContext = make_unique<MCBTFContext>();
}

void Dwarf2BTF::addTypeEntry(const DIE &Die) {
  for (auto &ChildDie : Die.children())
    addTypeEntry(ChildDie);
  if (Die2BTFEntry::shouldSkipDie(Die))
    return;
  auto Kind = Die2BTFEntry::getDieKind(Die);
  if (Kind != BTF_KIND_UNKN) {
    auto TypeEntry = Die2BTFEntry::dieToBTFTypeEntry(Die);
    if (TypeEntry != nullptr) {
      TypeEntry->setId(TypeEntries.size() + 1);
      DieToIdMap[const_cast<DIE*>(&Die)] = TypeEntry->getId();
      TypeEntries.push_back(std::move(TypeEntry));
    }
  }
}

void Dwarf2BTF::addBTFTypeEntry(std::unique_ptr<BTFTypeEntry> Entry) {
  BTFContext->addTypeEntry(std::move(Entry));
}

void Dwarf2BTF::completeData() {
  BTFContext->addString("\0");

  for (auto &TypeEntry : TypeEntries)
    TypeEntry->completeData(*this);
}

void Dwarf2BTF::addDwarfCU(DwarfUnit *TheU) {
  DIE &CuDie = TheU->getUnitDie();

  assert((CuDie.getTag() == dwarf::DW_TAG_compile_unit) &&
         "Not a compile unit");
  addTypeEntry(CuDie);
}

void Dwarf2BTF::finish() {
  completeData();
  OuterCtx.setBTFContext(std::move(BTFContext));
}

}

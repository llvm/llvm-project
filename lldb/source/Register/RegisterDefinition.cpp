//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Parsing and publication of external register images.
///
//===----------------------------------------------------------------------===//

#include "lldb/Register/RegisterDefinition.h"
#include "lldb/Utility/Format.h"
#include "lldb/Register/RegisterDescription.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/StringExtractor.h"
#include "lldb/Utility/StructuredData.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>
#include <iterator>

using namespace lldb;
using namespace lldb_private;

namespace {
/// Stage external definitions until both metadata and byte placement validate.
/// No partially built description or layout escapes through the public helpers.
class RegisterDefinitionParser {
public:
  using Register = RegisterDefinition;
  size_t BuildLayout(const StructuredData::Dictionary &dict,
                     const ArchSpec &arch);
  size_t BuildLayout(std::vector<Register> &&registers, const ArchSpec &arch,
                     RegisterDescription::OwnedTypes types = {});
  llvm::Error Publish(std::shared_ptr<const RegisterDescription> description,
                      llvm::ArrayRef<uint32_t> offsets,
                      lldb::ByteOrder byte_order);
  std::optional<RegisterLayout> layout;

private:
  void Clear() { layout.reset(); }
  size_t GetNumRegisters() const {
    return layout ? layout->GetDescription()->GetRegisters().size() : 0;
  }
  llvm::Expected<uint32_t> ByteOffsetFromSlice(std::vector<Register> &registers,
                                               uint32_t index,
                                               llvm::StringRef slice,
                                               lldb::ByteOrder byte_order);
  llvm::Expected<uint32_t>
  ByteOffsetFromComposite(std::vector<Register> &registers, uint32_t index,
                          StructuredData::Array &composite,
                          lldb::ByteOrder byte_order);
  llvm::Expected<uint32_t>
  ByteOffsetFromRegInfoDict(std::vector<Register> &registers, uint32_t index,
                            StructuredData::Dictionary &dict,
                            lldb::ByteOrder byte_order);
  size_t BuildLayout(std::vector<Register> &&registers, const ArchSpec &arch,
                     RegisterDescription::OwnedTypes types,
                     std::vector<std::string> groups);
};
} // namespace

static RegisterDefinition *
FindInputRegister(std::vector<RegisterDefinition> &registers,
                  llvm::StringRef name) {
  for (auto &reg : registers)
    if (reg.name.GetStringRef() == name)
      return &reg;
  return nullptr;
}

llvm::Expected<uint32_t> RegisterDefinitionParser::ByteOffsetFromSlice(
    std::vector<Register> &registers, uint32_t index, llvm::StringRef slice_str,
    lldb::ByteOrder byte_order) {
  // Slices use the following format:
  //  REGNAME[MSBIT:LSBIT]
  // REGNAME - name of the register to grab a slice of
  // MSBIT - the most significant bit at which the current register value
  // starts at
  // LSBIT - the least significant bit at which the current register value
  // ends at
  static llvm::Regex g_bitfield_regex(
      "([A-Za-z_][A-Za-z0-9_]*)\\[([0-9]+):([0-9]+)\\]");
  llvm::SmallVector<llvm::StringRef, 4> matches;
  if (!g_bitfield_regex.match(slice_str, &matches))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "failed to match against register bitfield regex (slice: %s)",
        slice_str.str().c_str());

  llvm::StringRef reg_name_str = matches[1];
  llvm::StringRef msbit_str = matches[2];
  llvm::StringRef lsbit_str = matches[3];
  uint32_t msbit;
  uint32_t lsbit;
  if (!llvm::to_integer(msbit_str, msbit) ||
      !llvm::to_integer(lsbit_str, lsbit))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(), "msbit (%s) or lsbit (%s) are invalid",
        msbit_str.str().c_str(), lsbit_str.str().c_str());

  if (msbit <= lsbit)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "msbit (%u) must be greater than lsbit (%u)",
                                   msbit, lsbit);

  const uint32_t msbyte = msbit / 8;
  const uint32_t lsbyte = lsbit / 8;

  const Register *containing_reg_info =
      FindInputRegister(registers, reg_name_str);
  if (!containing_reg_info)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid concrete register \"%s\"",
                                   reg_name_str.str().c_str());

  const uint32_t max_bit = containing_reg_info->byte_size * 8;

  if (msbit >= max_bit)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "msbit (%u) must be less than the bitsize of the register \"%s\" (%u)",
        msbit, reg_name_str.str().c_str(), max_bit);
  if (lsbit >= max_bit)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "lsbit (%u) must be less than the bitsize of the register \"%s\" (%u)",
        lsbit, reg_name_str.str().c_str(), max_bit);

  uint32_t parent = containing_reg_info - registers.data();
  registers[parent].invalidate_registers.push_back(index);
  registers[index].value_registers.push_back(parent);
  registers[index].invalidate_registers.push_back(parent);

  if (byte_order == eByteOrderLittle)
    return containing_reg_info->byte_offset + lsbyte;
  if (byte_order == eByteOrderBig)
    return containing_reg_info->byte_offset + containing_reg_info->byte_size -
           1 - msbyte;
  llvm_unreachable("Invalid byte order");
}

llvm::Expected<uint32_t> RegisterDefinitionParser::ByteOffsetFromComposite(
    std::vector<Register> &registers, uint32_t index,
    StructuredData::Array &composite_reg_list, lldb::ByteOrder byte_order) {
  const size_t num_composite_regs = composite_reg_list.GetSize();
  if (num_composite_regs == 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "\"composite\" list is empty");

  uint32_t composite_offset = UINT32_MAX;
  for (uint32_t composite_idx = 0; composite_idx < num_composite_regs;
       ++composite_idx) {
    std::optional<llvm::StringRef> maybe_composite_reg_name =
        composite_reg_list.GetItemAtIndexAsString(composite_idx);
    if (!maybe_composite_reg_name)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "\"composite\" list value is not a Python string at index %d",
          composite_idx);

    const Register *composite_reg_info =
        FindInputRegister(registers, *maybe_composite_reg_name);
    if (!composite_reg_info)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "failed to find composite register by name: \"%s\"",
          maybe_composite_reg_name->str().c_str());

    composite_offset =
        std::min(composite_offset, composite_reg_info->byte_offset);
    uint32_t parent = composite_reg_info - registers.data();
    registers[index].value_registers.push_back(parent);
    registers[parent].invalidate_registers.push_back(index);
    registers[index].invalidate_registers.push_back(parent);
  }

  return composite_offset;
}

llvm::Expected<uint32_t> RegisterDefinitionParser::ByteOffsetFromRegInfoDict(
    std::vector<Register> &registers, uint32_t index,
    StructuredData::Dictionary &reg_info_dict, lldb::ByteOrder byte_order) {
  uint32_t byte_offset;
  if (reg_info_dict.GetValueForKeyAsInteger("offset", byte_offset))
    return byte_offset;

  // No offset for this register, see if the register has a value
  // expression which indicates this register is part of another register.
  // Value expressions are things like "rax[31:0]" which state that the
  // current register's value is in a concrete register "rax" in bits 31:0.
  // If there is a value expression we can calculate the offset
  llvm::StringRef slice_str;
  if (reg_info_dict.GetValueForKeyAsString("slice", slice_str, nullptr))
    return ByteOffsetFromSlice(registers, index, slice_str, byte_order);

  StructuredData::Array *composite_reg_list;
  if (reg_info_dict.GetValueForKeyAsArray("composite", composite_reg_list))
    return ByteOffsetFromComposite(registers, index, *composite_reg_list,
                                   byte_order);

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "insufficient data to calculate byte offset");
}

size_t
RegisterDefinitionParser::BuildLayout(const StructuredData::Dictionary &dict,
                                      const ArchSpec &arch) {
  Log *log = GetLog(LLDBLog::Object);
  Clear();
  std::vector<Register> definitions;
  std::vector<std::string> group_names;
  StructuredData::Array *sets = nullptr;
  if (dict.GetValueForKeyAsArray("sets", sets)) {
    const uint32_t num_sets = sets->GetSize();
    for (uint32_t i = 0; i < num_sets; ++i) {
      std::optional<llvm::StringRef> maybe_set_name =
          sets->GetItemAtIndexAsString(i);
      if (maybe_set_name && !maybe_set_name->empty()) {
        group_names.push_back(maybe_set_name->str());
      } else {
        Clear();
        printf("error: register sets must have valid names\n");
        return 0;
      }
    }
  }

  StructuredData::Array *regs = nullptr;
  if (!dict.GetValueForKeyAsArray("registers", regs))
    return 0;

  const ByteOrder byte_order = arch.GetByteOrder();

  const uint32_t num_regs = regs->GetSize();
  definitions.reserve(num_regs);
  for (uint32_t i = 0; i < num_regs; ++i) {
    std::optional<StructuredData::Dictionary *> maybe_reg_info_dict =
        regs->GetItemAtIndexAsDictionary(i);
    if (!maybe_reg_info_dict) {
      Clear();
      printf("error: items in the 'registers' array must be dictionaries\n");
      regs->DumpToStdout();
      return 0;
    }
    StructuredData::Dictionary *reg_info_dict = *maybe_reg_info_dict;

    // { 'name':'rcx'       , 'bitsize' :  64, 'offset' :  16,
    // 'encoding':'uint' , 'format':'hex'         , 'set': 0, 'ehframe' : 2,
    // 'dwarf' : 2, 'generic':'arg4', 'alt-name':'arg4', },
    definitions.emplace_back();
    Register &reg_info = definitions.back();

    llvm::StringRef name_val;
    if (!reg_info_dict->GetValueForKeyAsString("name", name_val)) {
      Clear();
      printf("error: registers must have valid names and offsets\n");
      reg_info_dict->DumpToStdout();
      return 0;
    }
    reg_info.name = ConstString(name_val);

    llvm::StringRef alt_name_val;
    if (reg_info_dict->GetValueForKeyAsString("alt-name", alt_name_val))
      reg_info.alt_name = ConstString(alt_name_val);
    else
      reg_info.alt_name = ConstString();

    llvm::Expected<uint32_t> byte_offset =
        ByteOffsetFromRegInfoDict(definitions, i, *reg_info_dict, byte_order);
    if (byte_offset)
      reg_info.byte_offset = byte_offset.get();
    else {
      LLDB_LOG_ERROR(log, byte_offset.takeError(),
                     "error while parsing register {1}: {0}",
                     reg_info.name.AsCString(""));
      Clear();
      reg_info_dict->DumpToStdout();
      return 0;
    }

    uint64_t bitsize = 0;
    if (!reg_info_dict->GetValueForKeyAsInteger("bitsize", bitsize)) {
      Clear();
      printf("error: invalid or missing 'bitsize' key/value pair in register "
             "dictionary\n");
      reg_info_dict->DumpToStdout();
      return 0;
    }

    reg_info.byte_size = bitsize / 8;

    llvm::StringRef format_str;
    if (reg_info_dict->GetValueForKeyAsString("format", format_str, nullptr)) {
      if (!GetFormatFromCString(format_str.str().c_str(), reg_info.format)) {
        Clear();
        printf("error: invalid 'format' value in register dictionary\n");
        reg_info_dict->DumpToStdout();
        return 0;
      }
    } else {
      reg_info_dict->GetValueForKeyAsInteger("format", reg_info.format,
                                             eFormatHex);
    }

    llvm::StringRef encoding_str;
    if (reg_info_dict->GetValueForKeyAsString("encoding", encoding_str))
      reg_info.encoding = Args::StringToEncoding(encoding_str, eEncodingUint);
    else
      reg_info_dict->GetValueForKeyAsInteger("encoding", reg_info.encoding,
                                             eEncodingUint);

    size_t set = 0;
    if (!reg_info_dict->GetValueForKeyAsInteger("set", set) ||
        set >= group_names.size()) {
      Clear();
      printf("error: invalid 'set' value in register dictionary, valid values "
             "are 0 - %i\n",
             (int)set);
      reg_info_dict->DumpToStdout();
      return 0;
    }

    // Fill in the register numbers
    reg_info.id = i;
    reg_info.regnum_remote = i;
    reg_info.set_name = ConstString(group_names[set]);
    uint32_t eh_frame_regno = LLDB_INVALID_REGNUM;
    reg_info_dict->GetValueForKeyAsInteger("gcc", eh_frame_regno,
                                           LLDB_INVALID_REGNUM);
    if (eh_frame_regno == LLDB_INVALID_REGNUM)
      reg_info_dict->GetValueForKeyAsInteger("ehframe", eh_frame_regno,
                                             LLDB_INVALID_REGNUM);
    reg_info.regnum_ehframe = eh_frame_regno;
    reg_info_dict->GetValueForKeyAsInteger("dwarf", reg_info.regnum_dwarf,
                                           LLDB_INVALID_REGNUM);
    llvm::StringRef generic_str;
    if (reg_info_dict->GetValueForKeyAsString("generic", generic_str))
      reg_info.regnum_generic = Args::StringToGenericRegister(generic_str);
    else
      reg_info_dict->GetValueForKeyAsInteger("generic", reg_info.regnum_generic,
                                             LLDB_INVALID_REGNUM);

    // Check if this register invalidates any other register values when it is
    // modified
    StructuredData::Array *invalidate_reg_list = nullptr;
    if (reg_info_dict->GetValueForKeyAsArray("invalidate-regs",
                                             invalidate_reg_list)) {
      const size_t num_regs = invalidate_reg_list->GetSize();
      if (num_regs > 0) {
        for (uint32_t idx = 0; idx < num_regs; ++idx) {
          if (auto maybe_invalidate_reg_name =
                  invalidate_reg_list->GetItemAtIndexAsString(idx)) {
            const Register *invalidate_reg_info =
                FindInputRegister(definitions, *maybe_invalidate_reg_name);
            if (invalidate_reg_info) {
              reg_info.invalidate_registers.push_back(invalidate_reg_info -
                                                      definitions.data());
            } else {
              // TODO: print error invalid slice string that doesn't follow the
              // format
              printf("error: failed to find a 'invalidate-regs' register for "
                     "\"%s\" while parsing register \"%s\"\n",
                     maybe_invalidate_reg_name->str().c_str(),
                     reg_info.name.AsCString(""));
            }
          } else if (auto maybe_invalidate_reg_num =
                         invalidate_reg_list->GetItemAtIndexAsInteger<uint64_t>(
                             idx)) {
            if (*maybe_invalidate_reg_num != UINT64_MAX)
              reg_info.invalidate_registers.push_back(
                  *maybe_invalidate_reg_num);
            else
              printf("error: 'invalidate-regs' list value wasn't a valid "
                     "integer\n");
          } else {
            printf("error: 'invalidate-regs' list value wasn't a python string "
                   "or integer\n");
          }
        }
      } else {
        printf("error: 'invalidate-regs' contained an empty list\n");
      }
    }
  }
  return BuildLayout(std::move(definitions), arch, {}, std::move(group_names));
}

static void InferGenericRoles(std::vector<RegisterDescriptor> &registers,
                              const ArchSpec &arch) {
  // Check if we need to automatically set the generic registers in case they
  // weren't set
  bool generic_regs_specified = false;
  for (const auto &reg : registers) {
    if (reg.numbers[eRegisterKindGeneric] != LLDB_INVALID_REGNUM) {
      generic_regs_specified = true;
      break;
    }
  }

  if (!generic_regs_specified) {
    switch (arch.GetMachine()) {
    case llvm::Triple::aarch64:
    case llvm::Triple::aarch64_32:
    case llvm::Triple::aarch64_be:
      for (auto &reg : registers) {
        if (reg.name == "pc")
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC;
        else if ((reg.name == "fp") || (reg.name == "x29"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP;
        else if ((reg.name == "lr") || (reg.name == "x30"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_RA;
        else if ((reg.name == "sp") || (reg.name == "x31"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP;
        else if (reg.name == "cpsr")
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FLAGS;
      }
      break;

    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
      for (auto &reg : registers) {
        if ((reg.name == "pc") || (reg.name == "r15"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC;
        else if ((reg.name == "sp") || (reg.name == "r13"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP;
        else if ((reg.name == "lr") || (reg.name == "r14"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_RA;
        else if ((reg.name == "r7") &&
                 arch.GetTriple().getVendor() == llvm::Triple::Apple)
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP;
        else if ((reg.name == "r11") &&
                 arch.GetTriple().getVendor() != llvm::Triple::Apple)
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP;
        else if (reg.name == "fp")
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP;
        else if (reg.name == "cpsr")
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FLAGS;
      }
      break;

    case llvm::Triple::x86:
      for (auto &reg : registers) {
        if ((reg.name == "eip") || (reg.name == "pc"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC;
        else if ((reg.name == "esp") || (reg.name == "sp"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP;
        else if ((reg.name == "ebp") || (reg.name == "fp"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP;
        else if ((reg.name == "eflags") || (reg.name == "flags"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FLAGS;
      }
      break;

    case llvm::Triple::x86_64:
      for (auto &reg : registers) {
        if ((reg.name == "rip") || (reg.name == "pc"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC;
        else if ((reg.name == "rsp") || (reg.name == "sp"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP;
        else if ((reg.name == "rbp") || (reg.name == "fp"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP;
        else if ((reg.name == "rflags") || (reg.name == "eflags") ||
                 (reg.name == "flags"))
          reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FLAGS;
      }
      break;

    default:
      break;
    }
  }
}

void lldb_private::addSupplementaryRegister(
    std::vector<RegisterDefinition> &regs, RegisterDefinition new_reg_info) {
  assert(!new_reg_info.value_registers.empty());
  const uint32_t reg_num = regs.size();
  regs.push_back(new_reg_info);

  std::map<uint32_t, std::vector<uint32_t>> new_invalidates;
  for (uint32_t value_reg : new_reg_info.value_registers) {
    // copy value_regs to invalidate_regs
    new_invalidates[reg_num].push_back(value_reg);

    // copy invalidate_regs from the parent register
    llvm::append_range(new_invalidates[reg_num],
                       regs[value_reg].invalidate_registers);

    // add reverse invalidate entries
    for (uint32_t x : new_invalidates[reg_num])
      new_invalidates[x].push_back(reg_num);
  }

  for (const auto &x : new_invalidates)
    llvm::append_range(regs[x.first].invalidate_registers, x.second);
}

/// Assign packet byte offsets after logical relationships have been resolved.
/// Aliases use least-significant-byte offsets in metadata; translating them to
/// an image offset depends on byte order and belongs to this adapter.
static llvm::Expected<std::vector<uint32_t>>
PlaceRegisters(const RegisterDescription &description,
               llvm::ArrayRef<uint32_t> supplied, ByteOrder byte_order) {
  auto registers = description.GetRegisters();
  std::vector<uint32_t> offsets(supplied.begin(), supplied.end());
  // Some stubs repeat a remote number. Match remote relationship lookup by
  // using the last definition for that slot, without allocating extra bytes.
  std::map<uint32_t, size_t> order;
  for (size_t i = 0; i < registers.size(); ++i)
    if (registers[i].value_registers.empty())
      order[registers[i].numbers[eRegisterKindProcessPlugin]] = i;
  uint64_t cursor = 0;
  for (const auto &[number, i] : order) {
    if (offsets[i] == LLDB_INVALID_INDEX32) {
      if (cursor >= LLDB_INVALID_INDEX32)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "register image is too large");
      offsets[i] = cursor;
    }
    cursor = std::max(cursor, uint64_t(offsets[i]) + registers[i].byte_size);
    if (cursor >= LLDB_INVALID_INDEX32)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "register image is too large");
  }

  std::vector<uint8_t> state(registers.size(), 0);
  std::function<llvm::Error(size_t)> resolve = [&](size_t i) -> llvm::Error {
    if (state[i] == 2)
      return llvm::Error::success();
    if (state[i] == 1)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cyclic register aliases");
    state[i] = 1;
    const auto &reg = registers[i];
    if (offsets[i] == LLDB_INVALID_INDEX32 && reg.value_registers.empty()) {
      // Repeated definitions of a remote number name the same packet slot.
      // Keep earlier definitions readable, including a generic PC definition.
      offsets[i] = offsets[order.at(reg.numbers[eRegisterKindProcessPlugin])];
    } else if (offsets[i] == LLDB_INVALID_INDEX32 &&
               reg.value_registers.size() == 1) {
      size_t parent_index =
          *description.GetRegisterIndex(reg.value_registers.front());
      if (auto error = resolve(parent_index))
        return error;
      const auto &parent = registers[parent_index];
      if (reg.value_byte_offset > parent.byte_size ||
          reg.byte_size > parent.byte_size - reg.value_byte_offset)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "register alias exceeds its parent");
      if (offsets[parent_index] != LLDB_INVALID_INDEX32) {
        uint32_t offset =
            byte_order == eByteOrderBig
                ? parent.byte_size - reg.byte_size - reg.value_byte_offset
                : reg.value_byte_offset;
        offsets[i] = offsets[parent_index] + offset;
      }
    }
    state[i] = 2;
    return llvm::Error::success();
  };
  for (size_t i = 0; i < registers.size(); ++i)
    if (auto error = resolve(i))
      return std::move(error);
  return offsets;
}

size_t
RegisterDefinitionParser::BuildLayout(std::vector<Register> &&registers,
                                      const ArchSpec &arch,
                                      RegisterDescription::OwnedTypes types) {
  return BuildLayout(std::move(registers), arch, std::move(types), {});
}

size_t
RegisterDefinitionParser::BuildLayout(std::vector<Register> &&registers,
                                      const ArchSpec &arch,
                                      RegisterDescription::OwnedTypes types,
                                      std::vector<std::string> group_names) {
  Clear();
  std::vector<RegisterID> ids;
  for (auto [i, reg] : llvm::enumerate(registers))
    ids.push_back(reg.id == LLDB_INVALID_REGNUM ? i : reg.id);

  std::vector<RegisterDescriptor> values;
  std::vector<RegisterGroup> groups;
  for (auto &name : group_names)
    groups.push_back({std::move(name), {}, {}});
  std::vector<uint32_t> offsets;
  ByteOrder byte_order = arch.GetByteOrder();
  if (byte_order == eByteOrderInvalid)
    byte_order = eByteOrderLittle;
  auto build = [&]() -> llvm::Error {
    for (auto [i, reg] : llvm::enumerate(registers)) {
      RegisterDescriptor value(
          ids[i], reg.name.GetStringRef(), reg.alt_name.AsCString(nullptr),
          reg.byte_size, reg.encoding, reg.format,
          {reg.regnum_ehframe, reg.regnum_dwarf, reg.regnum_generic,
           reg.regnum_remote, static_cast<uint32_t>(i)},
          {}, {}, reg.type);
      for (auto [source, destination] :
           {std::make_pair(&reg.value_registers, &value.value_registers),
            std::make_pair(&reg.invalidate_registers,
                           &value.invalidate_registers)})
        for (uint32_t index : *source) {
          if (index >= ids.size())
            return llvm::createStringError(
                llvm::inconvertibleErrorCode(),
                "register reference is out of range");
          destination->push_back(ids[index]);
        }
      if (reg.value_registers.size() == 1) {
        const auto &parent = registers[reg.value_registers.front()];
        uint32_t offset = reg.value_reg_offset;
        if (reg.byte_offset != LLDB_INVALID_INDEX32 &&
            parent.byte_offset != LLDB_INVALID_INDEX32) {
          if (reg.byte_offset < parent.byte_offset)
            return llvm::createStringError(
                llvm::inconvertibleErrorCode(),
                "register alias precedes its parent");
          offset = reg.byte_offset - parent.byte_offset;
        }
        if (offset > parent.byte_size ||
            reg.byte_size > parent.byte_size - offset)
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "register alias exceeds its parent");
        value.value_byte_offset =
            byte_order == eByteOrderBig
                ? parent.byte_size - reg.byte_size - offset
                : offset;
      }
      values.push_back(std::move(value));
      offsets.push_back(reg.byte_offset);
      auto group = llvm::find_if(
          groups, [set_name = reg.set_name.GetStringRef()](const auto &candidate) {
            return candidate.name == set_name;
          });
      if (group == groups.end()) {
        groups.push_back({reg.set_name.GetStringRef().str(), {}, {}});
        group = std::prev(groups.end());
      }
      group->registers.push_back(ids[i]);
    }
    InferGenericRoles(values, arch);
    auto description = RegisterDescription::Create(
        std::move(values), std::move(groups), std::move(types));
    if (!description)
      return description.takeError();
    auto placed = PlaceRegisters(*description, offsets, byte_order);
    if (!placed)
      return placed.takeError();
    auto owned = std::make_shared<RegisterDescription>(std::move(*description));
    return Publish(std::move(owned), *placed, byte_order);
  };
  if (auto error = build()) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Object), std::move(error),
                   "invalid dynamic register definition: {0}");
    Clear();
    return 0;
  }
  return GetNumRegisters();
}

llvm::Error RegisterDefinitionParser::Publish(
    std::shared_ptr<const RegisterDescription> description,
    llvm::ArrayRef<uint32_t> offsets, ByteOrder byte_order) {
  RegisterLayout layout(byte_order);
  auto registers = description->GetRegisters();
  uint64_t byte_size = 0;
  for (auto [i, reg] : llvm::enumerate(registers))
    if (offsets[i] != LLDB_INVALID_INDEX32)
      byte_size = std::max(byte_size, uint64_t(offsets[i]) + reg.byte_size);
  if (byte_size >= LLDB_INVALID_INDEX32)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "register image is too large");
  if (auto error = layout.AddBank(0, byte_size))
    return error;

  // Composite values can be non-contiguous in the real image. Map their
  // constituent pieces instead of manufacturing a second serialized buffer.
  std::vector<uint8_t> state(registers.size(), 0);
  std::function<llvm::Error(size_t)> add = [&](size_t i) -> llvm::Error {
    if (state[i] == 2)
      return llvm::Error::success();
    if (state[i] == 1)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cyclic register aliases");
    state[i] = 1;
    const auto &reg = registers[i];
    RegisterLocation location;
    if (offsets[i] != LLDB_INVALID_INDEX32) {
      location.kind = RegisterLocation::Stored;
      location.pieces.push_back({0, offsets[i], reg.byte_size, 0});
    } else if (reg.value_registers.size() == 1) {
      RegisterID parent = reg.value_registers.front();
      size_t index = *description->GetRegisterIndex(parent);
      if (auto error = add(index))
        return error;
      const auto *parent_location = layout.Find(parent);
      if (parent_location->kind == RegisterLocation::Stored) {
        location.kind = RegisterLocation::Stored;
        location.zero_fill = parent_location->zero_fill;
        const auto &parent_reg = registers[index];
        uint64_t begin =
            byte_order == eByteOrderBig
                ? parent_reg.byte_size - reg.byte_size - reg.value_byte_offset
                : reg.value_byte_offset;
        uint64_t end = begin + reg.byte_size;
        for (const auto &piece : parent_location->pieces) {
          uint64_t first = std::max(begin, uint64_t(piece.value_offset));
          uint64_t last =
              std::min(end, uint64_t(piece.value_offset) + piece.byte_size);
          if (first < last)
            location.pieces.push_back(
                {piece.bank, piece.byte_offset + first - piece.value_offset,
                 static_cast<uint32_t>(last - first),
                 static_cast<uint32_t>(first - begin)});
        }
      }
    } else if (reg.value_registers.size() > 1) {
      location.kind = RegisterLocation::Stored;
      uint32_t value_offset = 0;
      for (RegisterID parent : reg.value_registers) {
        size_t index = *description->GetRegisterIndex(parent);
        if (auto error = add(index))
          return error;
        const auto *parent_location = layout.Find(parent);
        if (parent_location->kind != RegisterLocation::Stored)
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "unavailable composite register parent");
        for (auto piece : parent_location->pieces) {
          piece.value_offset += value_offset;
          location.pieces.push_back(piece);
        }
        value_offset += registers[index].byte_size;
      }
    }
    if (auto error = layout.Add(reg.id, std::move(location)))
      return error;
    state[i] = 2;
    return llvm::Error::success();
  };
  for (size_t i = 0; i < registers.size(); ++i)
    if (auto error = add(i))
      return error;
  auto finalized = RegisterLayout::Create(description, std::move(layout));
  if (!finalized)
    return finalized.takeError();
  this->layout.emplace(std::move(*finalized));
  return llvm::Error::success();
}

llvm::Expected<RegisterLayout> lldb_private::ReconfigureRegisterImageLayout(
    const RegisterLayout &layout,
    llvm::function_ref<void(llvm::MutableArrayRef<RegisterDescriptor>)>
        update) {
  auto old = layout.GetDescription();
  std::vector<RegisterDescriptor> registers(old->GetRegisters().begin(),
                                            old->GetRegisters().end());
  update(registers);
  auto description = RegisterDescription::Create(
      std::move(registers), {old->GetGroups().begin(), old->GetGroups().end()},
      {old->GetOwnedTypes().begin(), old->GetOwnedTypes().end()});
  if (!description)
    return description.takeError();
  std::vector<uint32_t> offsets(description->GetRegisters().size(),
                                LLDB_INVALID_INDEX32);
  ByteOrder byte_order = layout.GetByteOrder();
  auto placed = PlaceRegisters(*description, offsets, byte_order);
  if (!placed)
    return placed.takeError();
  RegisterDefinitionParser parser;
  if (auto error = parser.Publish(
          std::make_shared<RegisterDescription>(std::move(*description)),
          *placed, byte_order))
    return std::move(error);
  return std::move(*parser.layout);
}

std::shared_ptr<const RegisterLayout>
lldb_private::CreateRegisterImageLayout(const StructuredData::Dictionary &dict,
                                        const ArchSpec &arch) {
  RegisterDefinitionParser parser;
  parser.BuildLayout(dict, arch);
  if (!parser.layout)
    return nullptr;
  return std::make_shared<RegisterLayout>(std::move(*parser.layout));
}

std::shared_ptr<const RegisterLayout> lldb_private::CreateRegisterImageLayout(
    std::vector<RegisterDefinition> registers, const ArchSpec &arch,
    RegisterDescription::OwnedTypes types) {
  RegisterDefinitionParser parser;
  parser.BuildLayout(std::move(registers), arch, std::move(types));
  if (!parser.layout)
    return nullptr;
  return std::make_shared<RegisterLayout>(std::move(*parser.layout));
}

uint32_t lldb_private::GetRegisterImageOffset(const RegisterLayout &layout,
                                              RegisterID id) {
  const auto *reg = layout.GetDescription()->Find(id);
  const auto *location = layout.Find(id);
  if (!reg || !location || location->kind != RegisterLocation::Stored ||
      location->pieces.size() != 1 || location->pieces[0].bank != 0 ||
      location->pieces[0].value_offset != 0 ||
      location->pieces[0].byte_size != reg->byte_size)
    return LLDB_INVALID_INDEX32;
  return location->pieces[0].byte_offset;
}

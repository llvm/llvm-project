//===- offload-tblgen/RecordTypes.cpp - Offload record type wrappers -----===-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

#include "llvm/TableGen/Record.h"

namespace llvm {
namespace offload {
namespace tblgen {

class HandleRec {
public:
  explicit HandleRec(const Record *rec) : rec(rec) {}
  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }

private:
  const Record *rec;
};

class MacroRec {
public:
  explicit MacroRec(const Record *rec) : rec(rec) {
    auto Name = rec->getValueAsString("name");
    auto OpenBrace = Name.find_first_of("(");
    nameWithoutArgs = Name.substr(0, OpenBrace);
  }
  StringRef getName() const { return nameWithoutArgs; }
  StringRef getNameWithArgs() const { return rec->getValueAsString("name"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }

  std::optional<StringRef> getCondition() const {
    return rec->getValueAsOptionalString("condition");
  }
  StringRef getValue() const { return rec->getValueAsString("value"); }
  std::optional<StringRef> getAltValue() const {
    return rec->getValueAsOptionalString("alt_value");
  }

private:
  const Record *rec;
  std::string nameWithoutArgs;
};

class TypedefRec {
public:
  explicit TypedefRec(const Record *rec) : rec(rec) {}
  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }
  StringRef getValue() const { return rec->getValueAsString("value"); }

private:
  const Record *rec;
};

class EnumValueRec {
public:
  explicit EnumValueRec(const Record *rec) : rec(rec) {}
  std::string getName() const { return rec->getValueAsString("name").upper(); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }
  StringRef getTaggedType() const {
    return rec->getValueAsString("tagged_type");
  }

private:
  const Record *rec;
};

class EnumRec {
public:
  explicit EnumRec(const Record *rec) : rec(rec) {
    for (const auto *Val : rec->getValueAsListOfDefs("etors")) {
      vals.emplace_back(EnumValueRec{Val});
    }
  }
  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }
  const std::vector<EnumValueRec> &getValues() const { return vals; }

  std::string getEnumValNamePrefix() const {
    return StringRef(getName().str().substr(0, getName().str().length() - 2))
        .upper();
  }

  bool isTyped() const { return rec->getValueAsBit("is_typed"); }

  bool isBitField() const { return rec->getValueAsBit("is_bit_field"); }

private:
  const Record *rec;
  std::vector<EnumValueRec> vals;
};

class StructMemberRec {
public:
  explicit StructMemberRec(const Record *rec) : rec(rec) {}
  StringRef getType() const { return rec->getValueAsString("type"); }
  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }
  bool isPointerType() const { return getType().ends_with('*'); }
  bool isHandleType() const { return getType().ends_with("_handle_t"); }

private:
  const Record *rec;
};

class StructRec {
public:
  explicit StructRec(const Record *rec) : rec(rec) {
    for (auto *Member : rec->getValueAsListOfDefs("all_members")) {
      members.emplace_back(StructMemberRec(Member));
    }
  }
  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }
  std::optional<StringRef> getBaseClass() const {
    return rec->getValueAsOptionalString("base_class");
  }
  const std::vector<StructMemberRec> &getMembers() const { return members; }

private:
  const Record *rec;
  std::vector<StructMemberRec> members;
};

class ParamRec {
public:
  explicit ParamRec(const Record *rec) : rec(rec) {
    flags = rec->getValueAsBitsInit("flags");
    auto *Range = rec->getValueAsDef("range");
    auto RangeBegin = Range->getValueAsString("begin");
    auto RangeEnd = Range->getValueAsString("end");
    if (RangeBegin != "" && RangeEnd != "") {
      range = {RangeBegin, RangeEnd};
    } else {
      range = std::nullopt;
    }

    auto *TypeInfo = rec->getValueAsDef("type_info");
    auto TypeInfoEnum = TypeInfo->getValueAsString("enum");
    auto TypeInfoSize = TypeInfo->getValueAsString("size");
    if (TypeInfoEnum != "" && TypeInfoSize != "") {
      typeinfo = {TypeInfoEnum, TypeInfoSize};
    } else {
      typeinfo = std::nullopt;
    }
  }
  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getType() const { return rec->getValueAsString("type"); }
  bool isPointerType() const { return getType().ends_with('*'); }
  bool isHandleType() const { return getType().ends_with("_handle_t"); }
  bool isFptrType() const { return getType().ends_with("_cb_t"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }
  bool getFlagBit(unsigned int Bit) const {
    if (auto *BitValue = dyn_cast<BitInit>(flags->getBit(Bit)))
      return BitValue->getValue();
    assert(false && "Parameter flags has no default or set value");
    return false;
  }
  bool isIn() const { return getFlagBit(0); }
  bool isOut() const { return getFlagBit(1); }
  bool isOpt() const { return getFlagBit(2); }

  const Record *getRec() const { return rec; }
  std::optional<std::pair<StringRef, StringRef>> getRange() const {
    return range;
  }

  std::optional<std::pair<StringRef, StringRef>> getTypeInfo() const {
    return typeinfo;
  }

  // Needed to check whether we're at the back of a vector of params
  bool operator!=(const ParamRec &p) const { return rec != p.getRec(); }

private:
  const Record *rec;
  const BitsInit *flags;
  std::optional<std::pair<StringRef, StringRef>> range;
  std::optional<std::pair<StringRef, StringRef>> typeinfo;
};

class ReturnRec {
public:
  ReturnRec(const Record *rec) : rec(rec) {}
  StringRef getValue() const { return rec->getValueAsString("value"); }
  // Strip the "OL_ERRC_" from the value, resulting in just "FOO" from
  // "OL_ERRC_FOO"
  StringRef getUnprefixedValue() const {
    constexpr const char *ERRC = "ERRC_";
    auto Start = getValue().find(ERRC) + strlen(ERRC);
    return getValue().substr(Start);
  }
  std::vector<StringRef> getConditions() const {
    return rec->getValueAsListOfStrings("conditions");
  }

private:
  const Record *rec;
};

class FunctionRec {
public:
  FunctionRec(const Record *rec) : rec(rec) {
    for (auto &Ret : rec->getValueAsListOfDefs("all_returns"))
      rets.emplace_back(Ret);
    for (auto &Param : rec->getValueAsListOfDefs("params"))
      params.emplace_back(Param);
  }

  std::string getParamStructName() const {
    return llvm::formatv("{0}_params_t",
                         llvm::convertToSnakeFromCamelCase(getName()));
  }

  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getClass() const { return rec->getValueAsString("api_class"); }
  const std::vector<ReturnRec> &getReturns() const { return rets; }
  const std::vector<ParamRec> &getParams() const { return params; }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }
  std::vector<StringRef> getDetails() const {
    return rec->getValueAsListOfStrings("details");
  }
  std::vector<StringRef> getAnalogues() const {
    return rec->getValueAsListOfStrings("analogues");
  }

private:
  std::vector<ReturnRec> rets;
  std::vector<ParamRec> params;

  const Record *rec;
};

class FptrTypedefRec {
public:
  explicit FptrTypedefRec(const Record *rec) : rec(rec) {
    for (auto &Param : rec->getValueAsListOfDefs("params"))
      params.emplace_back(Param);
  }
  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }
  StringRef getReturn() const { return rec->getValueAsString("return"); }
  const std::vector<ParamRec> &getParams() const { return params; }

private:
  std::vector<ParamRec> params;

  const Record *rec;
};

} // namespace tblgen
} // namespace offload
} // namespace llvm

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
  explicit HandleRec(Record *rec) : rec(rec) {}
  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }

private:
  Record *rec;
};

class MacroRec {
public:
  explicit MacroRec(Record *rec) : rec(rec) {}
  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }

  std::optional<StringRef> getCondition() const {
    return rec->getValueAsOptionalString("condition");
  }
  StringRef getValue() const { return rec->getValueAsString("value"); }
  std::optional<StringRef> getAltValue() const {
    return rec->getValueAsOptionalString("alt_value");
  }

private:
  Record *rec;
};

class TypedefRec {
public:
  explicit TypedefRec(Record *rec) : rec(rec) {}
  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }
  StringRef getValue() const { return rec->getValueAsString("value"); }

private:
  Record *rec;
};

class EnumValueRec {
public:
  explicit EnumValueRec(Record *rec) : rec(rec) {}
  std::string getName() const { return rec->getValueAsString("name").upper(); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }

private:
  Record *rec;
};

class EnumRec {
public:
  explicit EnumRec(Record *rec) : rec(rec) {
    for (auto *Val : rec->getValueAsListOfDefs("etors")) {
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

private:
  Record *rec;
  std::vector<EnumValueRec> vals;
};

class StructMemberRec {
public:
  explicit StructMemberRec(Record *rec) : rec(rec) {}
  StringRef getType() const { return rec->getValueAsString("type"); }
  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }

private:
  Record *rec;
};

class StructRec {
public:
  explicit StructRec(Record *rec) : rec(rec) {
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
  Record *rec;
  std::vector<StructMemberRec> members;
};

class ParamRec {
public:
  explicit ParamRec(Record *rec) : rec(rec) {
    flags = rec->getValueAsBitsInit("flags");
  }
  StringRef getName() const { return rec->getValueAsString("name"); }
  StringRef getType() const { return rec->getValueAsString("type"); }
  StringRef getDesc() const { return rec->getValueAsString("desc"); }
  bool isIn() const { return dyn_cast<BitInit>(flags->getBit(0))->getValue(); }
  bool isOut() const { return dyn_cast<BitInit>(flags->getBit(1))->getValue(); }
  bool isOpt() const { return dyn_cast<BitInit>(flags->getBit(2))->getValue(); }

  Record *getRec() const { return rec; }

  // Needed to check whether we're at the back of a vector of params
  bool operator!=(const ParamRec &p) const { return rec != p.getRec(); }

private:
  Record *rec;
  BitsInit *flags;
};

class ReturnRec {
public:
  ReturnRec(Record *rec) : rec(rec) {}
  StringRef getValue() const { return rec->getValueAsString("value"); }
  std::vector<StringRef> getConditions() const {
    return rec->getValueAsListOfStrings("conditions");
  }

private:
  Record *rec;
};

class FunctionRec {
public:
  FunctionRec(Record *rec) : rec(rec) {
    for (auto &Ret : rec->getValueAsListOfDefs("all_returns"))
      rets.emplace_back(Ret);
    for (auto &Param : rec->getValueAsListOfDefs("params"))
      params.emplace_back(Param);
  }

  std::string getFullName() const {
    return rec->getValueAsString("api_class").str() +
           rec->getValueAsString("name").str();
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

  bool modifiesRefCount() const {
    auto Name = rec->getValueAsString("name");
    auto Class = rec->getValueAsString("api_class");
    return (Name == "Create") || (Name == "Retain") || (Name == "Release") ||
           (Name == "Get" && Class == "Adapter");
  }

private:
  std::vector<ReturnRec> rets;
  std::vector<ParamRec> params;

  Record *rec;
};

} // namespace tblgen
} // namespace offload
} // namespace llvm

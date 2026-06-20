
#ifndef LLVM_TOOLS_LLVM_ADVISOR_REMARKSRELATIONALSCHEMA_H
#define LLVM_TOOLS_LLVM_ADVISOR_REMARKSRELATIONALSCHEMA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <string>
#include <vector>

namespace llvm {
namespace advisor {

class RelationalStringTable {
public:
  unsigned getOrAdd(StringRef S) {
    auto [It, Inserted] = Index.try_emplace(S, Strings.size());
    if (Inserted)
      Strings.emplace_back(S.str());
    return It->second;
  }

  StringRef get(int64_t Idx) const {
    if (Idx < 0 || static_cast<size_t>(Idx) >= Strings.size())
      return "";
    return Strings[static_cast<size_t>(Idx)];
  }

  json::Array toJSON() const {
    json::Array Out;
    Out.reserve(Strings.size());
    for (const std::string &S : Strings)
      Out.push_back(S);
    return Out;
  }

  void writeJSON(json::OStream &JOS) const {
    JOS.arrayBegin();
    for (const std::string &S : Strings)
      JOS.value(S);
    JOS.arrayEnd();
  }

private:
  std::vector<std::string> Strings;
  StringMap<unsigned> Index;
};

inline void writeInt64Column(json::OStream &JOS, StringRef Name,
                             ArrayRef<int64_t> Vs) {
  JOS.attributeBegin(Name);
  JOS.arrayBegin();
  for (int64_t V : Vs)
    JOS.value(V);
  JOS.arrayEnd();
  JOS.attributeEnd();
}

} // namespace advisor
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_ADVISOR_REMARKSRELATIONALSCHEMA_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypePool.h"
#include "llvm/ADT/SmallVector.h"
#include <cstring>

using namespace llvm;
using namespace dwarf_linker;
using namespace dwarf_linker::parallel;

void TypeKey::appendRef(const TypeEntry *Entry) {
  assert(Entry && "null reference");
  Bytes.push_back(RefMarker);
  const char *EntryBytes = reinterpret_cast<const char *>(&Entry);
  Bytes.append(EntryBytes, EntryBytes + sizeof(Entry));
}

TypeKey::Part TypeKey::takePart(StringRef &Key) {
  assert(!Key.empty() && "no part to take");
  if (Key.front() != RefMarker) {
    StringRef Text = Key.take_front(Key.find(RefMarker));
    Key = Key.drop_front(Text.size());
    return Text;
  }
  assert(Key.size() >= RefSize && "truncated reference");
  const TypeEntry *Ref;
  std::memcpy(&Ref, Key.data() + 1, sizeof(Ref));
  Key = Key.drop_front(RefSize);
  return Ref;
}

namespace {
/// Walks a type key as if every reference were replaced by its key.
class ExpandedKeyCursor {
public:
  ExpandedKeyCursor(StringRef Key) { Pending.push_back(Key); }

  bool atEnd() {
    if (const auto *Text = std::get_if<StringRef>(&Head);
        !Text || !Text->empty())
      return false;
    while (!Pending.empty() && Pending.back().empty())
      Pending.pop_back();
    if (Pending.empty())
      return true;
    Head = TypeKey::takePart(Pending.back());
    return false;
  }

  /// \pre !atEnd()
  const TypeKey::Part &peek() const { return Head; }

  /// \pre peek() is text of at least \p N bytes.
  void consumeText(size_t N) {
    StringRef &Text = std::get<StringRef>(Head);
    Text = Text.drop_front(N);
  }

  /// \pre peek() is a reference.
  void skipRef() { Head = {}; }

  /// Replace the reference at the cursor by the key it refers to.
  /// \pre peek() is a reference.
  void enterRef() {
    Pending.push_back(std::get<const TypeEntry *>(Head)->getKey());
    Head = {};
  }

private:
  /// The undecoded rest of each key being expanded, innermost last.
  SmallVector<StringRef, 8> Pending;
  /// Decoded once, so consuming it piecemeal does not rescan it. Empty text
  /// until the next part is decoded.
  TypeKey::Part Head;
};
} // namespace

/// Order keys by their text runs and references, recursively. Unlike
/// expansion this is a total order on distinct keys.
static int compareTypeKeyStructure(StringRef LHS, StringRef RHS) {
  while (!LHS.empty() && !RHS.empty()) {
    TypeKey::Part L = TypeKey::takePart(LHS);
    TypeKey::Part R = TypeKey::takePart(RHS);
    if (L.index() != R.index())
      return std::holds_alternative<const TypeEntry *>(L) ? -1 : 1;
    if (L == R)
      continue;
    if (const auto *LText = std::get_if<StringRef>(&L))
      return LText->compare(std::get<StringRef>(R));
    if (int Cmp =
            compareTypeKeyStructure(std::get<const TypeEntry *>(L)->getKey(),
                                    std::get<const TypeEntry *>(R)->getKey()))
      return Cmp;
  }
  return LHS.empty() == RHS.empty() ? 0 : (LHS.empty() ? -1 : 1);
}

int llvm::dwarf_linker::parallel::compareTypeKeys(StringRef LHS,
                                                  StringRef RHS) {
  ExpandedKeyCursor L(LHS), R(RHS);
  while (true) {
    bool LEnd = L.atEnd();
    bool REnd = R.atEnd();
    // Distinct keys expand equally if a name contains key syntax. Break the
    // tie without depending on pointer values.
    if (LEnd && REnd)
      return LHS == RHS ? 0 : compareTypeKeyStructure(LHS, RHS);
    if (LEnd || REnd)
      return LEnd ? -1 : 1;

    const TypeKey::Part &LPart = L.peek();
    const TypeKey::Part &RPart = R.peek();
    bool LRef = std::holds_alternative<const TypeEntry *>(LPart);
    bool RRef = std::holds_alternative<const TypeEntry *>(RPart);
    // Equal references refer to equal keys, so they need no expanding.
    if (LRef && LPart == RPart) {
      L.skipRef();
      R.skipRef();
      continue;
    }
    if (LRef || RRef) {
      if (LRef)
        L.enterRef();
      if (RRef)
        R.enterRef();
      continue;
    }

    StringRef LText = std::get<StringRef>(LPart);
    StringRef RText = std::get<StringRef>(RPart);
    size_t N = std::min(LText.size(), RText.size());
    if (int Cmp = LText.take_front(N).compare(RText.take_front(N)))
      return Cmp;
    L.consumeText(N);
    R.consumeText(N);
  }
}

std::string
llvm::dwarf_linker::parallel::getExpandedTypeKey(const TypeEntry &Entry) {
  std::string Result;
  ExpandedKeyCursor Cursor(Entry.getKey());
  while (!Cursor.atEnd()) {
    const auto *Text = std::get_if<StringRef>(&Cursor.peek());
    if (!Text) {
      Cursor.enterRef();
      continue;
    }
    Result += *Text;
    Cursor.consumeText(Text->size());
  }
  return Result;
}

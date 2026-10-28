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
  Part Result;
  if (Key.front() != RefMarker) {
    Result.Text = Key.take_front(Key.find(RefMarker));
    Key = Key.drop_front(Result.Text.size());
    return Result;
  }
  assert(Key.size() >= RefSize && "truncated reference");
  std::memcpy(&Result.Ref, Key.data() + 1, sizeof(Result.Ref));
  Key = Key.drop_front(RefSize);
  return Result;
}

namespace {
/// Walks a type key as if every reference were replaced by its key.
class ExpandedKeyCursor {
public:
  ExpandedKeyCursor(StringRef Key) { Pending.push_back(Key); }

  bool atEnd() {
    if (!Head.Text.empty() || Head.Ref)
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
  void consumeText(size_t N) { Head.Text = Head.Text.drop_front(N); }

  /// \pre peek() is a reference.
  void skipRef() { Head = {}; }

  /// Replace the reference at the cursor by the key it refers to.
  /// \pre peek() is a reference.
  void enterRef() {
    Pending.push_back(Head.Ref->getKey());
    Head = {};
  }

private:
  /// The undecoded rest of each key being expanded, innermost last.
  SmallVector<StringRef, 8> Pending;
  /// Decoded once, so consuming it piecemeal does not rescan it. Empty until
  /// the next part is decoded.
  TypeKey::Part Head;
};
} // namespace

/// Order keys by their text runs and references, recursively. Unlike
/// expansion this is a total order on distinct keys.
static int compareTypeKeyStructure(StringRef LHS, StringRef RHS) {
  while (!LHS.empty() && !RHS.empty()) {
    TypeKey::Part L = TypeKey::takePart(LHS);
    TypeKey::Part R = TypeKey::takePart(RHS);
    bool LRef = L.Ref != nullptr;
    bool RRef = R.Ref != nullptr;
    if (LRef != RRef)
      return LRef ? -1 : 1;
    if (LRef) {
      if (L.Ref != R.Ref)
        if (int Cmp = compareTypeKeyStructure(L.Ref->getKey(), R.Ref->getKey()))
          return Cmp;
      continue;
    }
    if (int Cmp = L.Text.compare(R.Text))
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

    TypeKey::Part LPart = L.peek();
    TypeKey::Part RPart = R.peek();
    // Equal references refer to equal keys, so they need no expanding.
    if (LPart.Ref && LPart.Ref == RPart.Ref) {
      L.skipRef();
      R.skipRef();
      continue;
    }
    if (LPart.Ref || RPart.Ref) {
      if (LPart.Ref)
        L.enterRef();
      if (RPart.Ref)
        R.enterRef();
      continue;
    }

    size_t N = std::min(LPart.Text.size(), RPart.Text.size());
    if (int Cmp = LPart.Text.take_front(N).compare(RPart.Text.take_front(N)))
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
    TypeKey::Part Part = Cursor.peek();
    if (Part.Ref) {
      Cursor.enterRef();
      continue;
    }
    Result += Part.Text;
    Cursor.consumeText(Part.Text.size());
  }
  return Result;
}

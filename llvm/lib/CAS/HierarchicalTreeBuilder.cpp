//===- HierarchicalTreeBuilder.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/Utils.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::cas;

/// Critical to canonicalize components so that paths come up next to each
/// other when sorted.
static StringRef canonicalize(SmallVectorImpl<char> &Path,
                              TreeEntry::EntryKind Kind) {
  // Make absolute.
  if (Path.empty() || Path.front() != '/')
    Path.insert(Path.begin(), '/');

  // FIXME: consider rejecting ".." instead of removing them.
  sys::path::remove_dots(Path, /*remove_dot_dot=*/true,
                         sys::path::Style::posix);

  // Canonicalize slashes.
  bool PendingSlash = false;
  char *NewEnd = Path.begin();
  for (int I = 0, E = Path.size(); I != E; ++I) {
    if (Path[I] == '/') {
      PendingSlash = true;
      continue;
    }
    if (PendingSlash)
      *NewEnd++ = '/';
    PendingSlash = false;
    *NewEnd++ = Path[I];
  }
  Path.erase(NewEnd, Path.end());

  // For correct sorting, all explicit trees need to end with a '/'.
  if (Path.empty() || Kind == TreeEntry::Tree)
    Path.push_back('/');
  return StringRef(Path.begin(), Path.size());
}

void HierarchicalTreeBuilder::pushImpl(Optional<ObjectRef> Ref,
                                       TreeEntry::EntryKind Kind,
                                       const Twine &Path) {
  SmallVector<char, 256> CanonicalPath;
  Path.toVector(CanonicalPath);
  Entries.emplace_back(Ref, Kind, canonicalize(CanonicalPath, Kind));
}

void HierarchicalTreeBuilder::pushTreeContent(ObjectRef Ref,
                                              const Twine &Path) {
  SmallVector<char, 256> CanonicalPath;
  Path.toVector(CanonicalPath);
  TreeEntry::EntryKind Kind = TreeEntry::Tree;
  TreeContents.emplace_back(Ref, Kind, canonicalize(CanonicalPath, Kind));
}

Expected<ObjectHandle> HierarchicalTreeBuilder::create(CASDB &CAS) {
  // FIXME: It is inefficient expanding the whole tree recursively like this,
  // use a more efficient algorithm to merge contents.
  TreeSchema Schema(CAS);
  for (const auto &TreeContent : TreeContents) {
    Optional<ObjectHandle> LoadedTree;
    if (Error E = CAS.load(*TreeContent.getRef()).moveInto(LoadedTree))
      return std::move(E);
    StringRef Path = TreeContent.getPath();
    Error E = Schema.walkFileTreeRecursively(
        CAS, *LoadedTree,
        [&](const NamedTreeEntry &Entry,
            Optional<TreeProxy> Tree) -> Error {
          if (Entry.getKind() != TreeEntry::Tree) {
            pushImpl(Entry.getRef(), Entry.getKind(), Path + Entry.getName());
            return Error::success();
          }
          if (Tree->empty())
            pushDirectory(Path + Entry.getName());
          return Error::success();
        });
    if (E)
      return std::move(E);
  }
  TreeContents.clear();

  if (Entries.empty())
    return Schema.create();

  std::stable_sort(
      Entries.begin(), Entries.end(),
      [](const HierarchicalEntry &LHS, const HierarchicalEntry &RHS) {
        // Lexicographically smaller paths first.
        if (int Compare = LHS.getPath().compare(RHS.getPath()))
          return Compare < 0;

        // Nodes with IDs first (only trees may have a missing Ref).
        return bool(LHS.getRef()) > bool(RHS.getRef());
      });

  // Compile into trees.
  struct Tree;
  struct Node {
    Node *Next = nullptr;
    Tree *Parent = nullptr;
    Optional<ObjectRef> Ref;
    TreeEntry::EntryKind Kind;
    StringRef Name;

    bool isTree() const { return Kind == TreeEntry::Tree; }
  };
  struct Tree : Node {
    Node *First = nullptr;
    bool Visited = false;
  };

  BumpPtrAllocator Alloc;
  Tree Root;
  const HierarchicalEntry *PrevEntry = nullptr;
  for (const HierarchicalEntry &Entry : Entries) {
    // Check for duplicates.
    if (PrevEntry && PrevEntry->getPath() == Entry.getPath()) {
      // Error if it's not identical.
      //
      // FIXME: Maybe we should allow clobbering / merging / etc., but for now
      // just error.
      if (Entry.getKind() != PrevEntry->getKind())
        return createStringError(
            std::make_error_code(std::errc::invalid_argument),
            "duplicate path '" + Entry.getPath() + "' with different kind");
      if (!Entry.getRef()) {
        assert(Entry.getKind() == TreeEntry::Tree);
        continue;
      }
      assert(PrevEntry->getRef());
      if (*Entry.getRef() != *PrevEntry->getRef())
        return createStringError(
            std::make_error_code(std::errc::invalid_argument),
            "duplicate path '" + Entry.getPath() + "' with different ID");

      // Skip the duplicate.
      continue;
    }
    PrevEntry = &Entry;

    Tree *Current = &Root;
    StringRef Path = Entry.getPath();
    {
      bool Consumed = Path.consume_front("/");
      (void)Consumed;
      assert(Consumed && "Expected canonical POSIX absolute paths");
    }
    for (auto Slash = Path.find('/'); !Path.empty(); Slash = Path.find('/')) {
      StringRef Name;
      if (Slash == StringRef::npos) {
        Name = Path;
        Path = "";
      } else {
        Name = Path.take_front(Slash);
        Path = Path.drop_front(Slash + 1);
      }

      // If the tree Current already has a ref, then it's fixed and we can't
      // add anything to it.
      if (Current->Ref)
        return createStringError(
            std::make_error_code(std::errc::invalid_argument),
            "cannot add '" + Entry.getPath() + "' under fixed tree");

      // Need to canonicalize first, or else the sorting trick doesn't work.
      assert(Name != "");
      assert(Name != "/");
      assert(Name != ".");
      assert(Name != "..");

      // Check if it's the first node (sorting ahead of time means it's either
      // the first node, or it doesn't exist yet). Also, check for conflicts
      // between implied trees and other nodes, such as a blob "/a" and an
      // implied tree from "/a/b".
      if (Current->First && Name == Current->First->Name) {
        if (Path == "" && Entry.getKind() == TreeEntry::Tree) {
          // Tree already exists. Sort order should ensure a fixed tree comes
          // first.
          assert(!Entry.getRef() ||
                 (Current->Ref && *Current->Ref == *Entry.getRef()));
          break;
        }
        if (Current->First->Kind == TreeEntry::Tree) {
          // Navigate deeper.
          Current = static_cast<Tree *>(Current->First);
          continue;
        }

        // This is reachable if there are two entries "/duplicate" and
        // "/duplicate/suffix".
        return createStringError(
            std::make_error_code(std::errc::invalid_argument),
            "duplicate path '" +
                Entry.getPath().take_front(Name.end() -
                                           Entry.getPath().begin()) +
                "'");
      }

      // Doesn't exist yet.
      Node *New;
      Tree *Next = nullptr;
      if (Path == "" && Entry.getKind() != TreeEntry::Tree) {
        New = new (Alloc.Allocate<Node>()) Node();
      } else {
        Next = new (Alloc.Allocate<Tree>()) Tree();
        New = Next;
      }
      New->Parent = Current;
      New->Next = Current->First;
      New->Name = Name;
      if (Path == "") {
        New->Kind = Entry.getKind();
        New->Ref = Entry.getRef();
      } else {
        New->Kind = TreeEntry::Tree;
      }
      Current->First = New;
      Current = Next;
    }
  }

  // Create the trees bottom up. Pre-allocate space for 8 entries, since many
  // trees are fairly small when building cache keys.
  SmallVector<NamedTreeEntry, 8> Entries;
  SmallVector<Tree *> Worklist = {&Root};
  while (!Worklist.empty()) {
    Tree *T = Worklist.back();
    if (!T->Visited) {
      assert(!T->Ref && "Trees with fixed content shouldn't be visited");
      for (Node *N = T->First; N; N = N->Next) {
        if (!N->Ref) {
          assert(N->Kind == TreeEntry::Tree);
          Worklist.push_back(static_cast<Tree *>(N));
        }
      }
      T->Visited = true;
      continue;
    }

    Worklist.pop_back();
    for (Node *N = T->First; N; N = N->Next)
      Entries.emplace_back(*N->Ref, N->Kind, N->Name);
    Expected<TreeProxy> ExpectedTree = Schema.create(Entries);
    Entries.clear();
    if (!ExpectedTree)
      return ExpectedTree.takeError();
    T->Ref = ExpectedTree->getRef();
  }

  return cantFail(CAS.load(*Root.Ref));
}

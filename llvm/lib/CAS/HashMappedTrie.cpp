//===- HashMappedTrie.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/HashMappedTrie.h"
#include "HashMappedTrieIndexGenerator.h"
#include "llvm/ADT/LazyAtomicPointer.h"
#include "llvm/CAS/ThreadSafeAllocator.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace llvm;
using namespace llvm::cas;

namespace {
struct TrieNode {
  const bool IsSubtrie = false;

  TrieNode(bool IsSubtrie) : IsSubtrie(IsSubtrie) {}

  static void *operator new(size_t Size) { return ::malloc(Size); }
  void operator delete(void *Ptr) { ::free(Ptr); }
};

struct TrieContent final : public TrieNode {
  const uint8_t ContentOffset;
  const uint8_t HashSize;
  const uint8_t HashOffset;

  void *getValuePointer() const {
    auto Content = reinterpret_cast<const uint8_t *>(this) + ContentOffset;
    return const_cast<uint8_t *>(Content);
  }

  ArrayRef<uint8_t> getHash() const {
    auto *Begin = reinterpret_cast<const uint8_t *>(this) + HashOffset;
    return makeArrayRef(Begin, Begin + HashSize);
  }

  TrieContent(size_t ContentOffset, size_t HashSize, size_t HashOffset)
      : TrieNode(/*IsSubtrie=*/false), ContentOffset(ContentOffset),
        HashSize(HashSize), HashOffset(HashOffset) {}
};
static_assert(sizeof(TrieContent) ==
                  ThreadSafeHashMappedTrieBase::TrieContentBaseSize,
              "Check header assumption!");

class TrieSubtrie final : public TrieNode {
public:
  TrieNode *get(size_t I) const { return Slots[I].load(); }

  TrieSubtrie *
  sink(size_t I, TrieContent &Content, size_t NumSubtrieBits, size_t NewI,
       function_ref<TrieSubtrie *(std::unique_ptr<TrieSubtrie>)> Saver);

  void printHash(raw_ostream &OS, ArrayRef<uint8_t> Bytes) const;
  void print(raw_ostream &OS) const { print(OS, None); }
  void print(raw_ostream &OS, Optional<std::string> Prefix) const;
  void dump() const { print(dbgs()); }

  static std::unique_ptr<TrieSubtrie> create(size_t StartBit, size_t NumBits);

  explicit TrieSubtrie(size_t StartBit, size_t NumBits);

private:
  // FIXME: Use a bitset to speed up access:
  //
  //     std::array<std::atomic<uint64_t>, NumSlots/64> IsSet;
  //
  // This will avoid needing to visit sparsely filled slots in
  // \a ThreadSafeHashMappedTrieBase::destroyImpl() when there's a non-trivial
  // destructor.
  //
  // It would also greatly speed up iteration, if we add that some day, and
  // allow get() to return one level sooner.
  //
  // This would be the algorithm for updating IsSet (after updating Slots):
  //
  //     std::atomic<uint64_t> &Bits = IsSet[I.High];
  //     const uint64_t NewBit = 1ULL << I.Low;
  //     uint64_t Old = 0;
  //     while (!Bits.compare_exchange_weak(Old, Old | NewBit))
  //       ;

  // For debugging.
  unsigned StartBit = 0;
  unsigned NumBits = 0;

public:
  /// Linked list for ownership of tries. The pointer is owned by TrieSubtrie.
  std::atomic<TrieSubtrie *> Next;

  /// The (co-allocated) slots of the subtrie.
  MutableArrayRef<LazyAtomicPointer<TrieNode>> Slots;
};
} // end namespace

namespace llvm {
template <> struct isa_impl<TrieContent, TrieNode> {
  static inline bool doit(const TrieNode &TN) { return !TN.IsSubtrie; }
};
template <> struct isa_impl<TrieSubtrie, TrieNode> {
  static inline bool doit(const TrieNode &TN) { return TN.IsSubtrie; }
};
} // end namespace llvm

static size_t getTrieTailSize(size_t StartBit, size_t NumBits) {
  assert(NumBits < 20 && "Tries should have fewer than ~1M slots");
  return sizeof(TrieNode *) * (1u << NumBits);
}

std::unique_ptr<TrieSubtrie> TrieSubtrie::create(size_t StartBit,
                                                 size_t NumBits) {
  size_t Size = sizeof(TrieSubtrie) + getTrieTailSize(StartBit, NumBits);
  void *Memory = ::malloc(Size);
  TrieSubtrie *S = ::new (Memory) TrieSubtrie(StartBit, NumBits);
  return std::unique_ptr<TrieSubtrie>(S);
}

TrieSubtrie::TrieSubtrie(size_t StartBit, size_t NumBits)
    : TrieNode(true), StartBit(StartBit), NumBits(NumBits), Next(nullptr),
      Slots(reinterpret_cast<LazyAtomicPointer<TrieNode> *>(
                reinterpret_cast<char *>(this) + sizeof(TrieSubtrie)),
            (1u << NumBits)) {
  for (auto *I = Slots.begin(), *E = Slots.end(); I != E; ++I)
    new (I) LazyAtomicPointer<TrieNode>(nullptr);

  static_assert(
      std::is_trivially_destructible<LazyAtomicPointer<TrieNode>>::value,
      "Expected no work in destructor for TrieNode");
}

TrieSubtrie *TrieSubtrie::sink(
    size_t I, TrieContent &Content, size_t NumSubtrieBits, size_t NewI,
    function_ref<TrieSubtrie *(std::unique_ptr<TrieSubtrie>)> Saver) {
  assert(NumSubtrieBits > 0);
  std::unique_ptr<TrieSubtrie> S = create(StartBit + NumBits, NumSubtrieBits);

  assert(NewI < S->Slots.size());
  S->Slots[NewI].store(&Content);

  TrieNode *ExistingNode = &Content;
  assert(I < Slots.size());
  if (Slots[I].compare_exchange_strong(ExistingNode, S.get()))
    return Saver(std::move(S));

  // Another thread created a subtrie already. Return it and let "S" be
  // destructed.
  return cast<TrieSubtrie>(ExistingNode);
}

struct ThreadSafeHashMappedTrieBase::ImplType {
  static ImplType *create(size_t StartBit, size_t NumBits) {
    size_t Size = sizeof(ImplType) + getTrieTailSize(StartBit, NumBits);
    void *Memory = ::malloc(Size);
    return ::new (Memory) ImplType(StartBit, NumBits);
  }

  TrieSubtrie *save(std::unique_ptr<TrieSubtrie> S) {
    assert(!S->Next && "Expected S to a freshly-constructed leaf");

    TrieSubtrie *CurrentHead = nullptr;
    // Add ownership of "S" to front of the list, so that Root -> S ->
    // Root.Next. This works by repeatedly setting S->Next to a candidate value
    // of Root.Next (initially nullptr), then setting Root.Next to S once the
    // candidate matches reality.
    while (!Root.Next.compare_exchange_weak(CurrentHead, S.get()))
      S->Next.exchange(CurrentHead);

    // Ownership transferred to subtrie.
    return S.release();
  }

  /// FIXME: This should take a function that allocates and constructs the
  /// content lazily (taking the hash as a separate parameter), in case of
  /// collision.
  ThreadSafeAllocator<BumpPtrAllocator> ContentAlloc;
  TrieSubtrie Root; // Must be last! Tail-allocated.

private:
  ImplType(size_t StartBit, size_t NumBits) : Root(StartBit, NumBits) {}
};

ThreadSafeHashMappedTrieBase::ImplType &
ThreadSafeHashMappedTrieBase::getOrCreateImpl() {
  if (ImplType *Impl = ImplPtr.load())
    return *Impl;

  // Create a new ImplType and store it if another thread doesn't do so first.
  // If another thread wins this one is destroyed locally.
  std::unique_ptr<ImplType> Impl(ImplType::create(0, NumRootBits));
  ImplType *ExistingImpl = nullptr;
  if (ImplPtr.compare_exchange_strong(ExistingImpl, Impl.get()))
    return *Impl.release();

  return *ExistingImpl;
}

ThreadSafeHashMappedTrieBase::PointerBase
ThreadSafeHashMappedTrieBase::find(ArrayRef<uint8_t> Hash) const {
  assert(!Hash.empty() && "Uninitialized hash");

  ImplType *Impl = ImplPtr.load();
  if (!Impl)
    return PointerBase();

  TrieSubtrie *S = &Impl->Root;
  IndexGenerator IndexGen{NumRootBits, NumSubtrieBits, Hash};
  size_t Index = IndexGen.next();
  for (;;) {
    // Try to set the content.
    TrieNode *Existing = S->get(Index);
    if (!Existing)
      return PointerBase(S, Index, *IndexGen.StartBit);

    // Check for an exact match.
    if (auto *ExistingContent = dyn_cast<TrieContent>(Existing))
      return ExistingContent->getHash() == Hash
                 ? PointerBase(ExistingContent->getValuePointer())
                 : PointerBase(S, Index, *IndexGen.StartBit);

    Index = IndexGen.next();
    S = cast<TrieSubtrie>(Existing);
  }
}

ThreadSafeHashMappedTrieBase::PointerBase ThreadSafeHashMappedTrieBase::insert(
    PointerBase Hint, ArrayRef<uint8_t> Hash,
    function_ref<const uint8_t *(void *Mem, ArrayRef<uint8_t> Hash)>
        Constructor) {
  assert(!Hash.empty() && "Uninitialized hash");

  ImplType &Impl = getOrCreateImpl();
  TrieSubtrie *S = &Impl.Root;
  IndexGenerator IndexGen{NumRootBits, NumSubtrieBits, Hash};
  size_t Index;
  if (Hint.isHint()) {
    S = static_cast<TrieSubtrie *>(Hint.P);
    Index = IndexGen.hint(Hint.I, Hint.B);
  } else {
    Index = IndexGen.next();
  }

  for (;;) {
    // Load the node from the slot, allocating and calling the constructor if
    // the slot is empty.
    bool Generated = false;
    TrieNode &Existing = S->Slots[Index].loadOrGenerate([&]() {
      Generated = true;

      // Construct the value itself at the tail.
      uint8_t *Memory = reinterpret_cast<uint8_t *>(
          Impl.ContentAlloc.Allocate(ContentAllocSize, ContentAllocAlign));
      const uint8_t *HashStorage = Constructor(Memory + ContentOffset, Hash);

      // Construct the TrieContent header, passing in the offset to the hash.
      TrieContent *Content = ::new (Memory)
          TrieContent(ContentOffset, Hash.size(), HashStorage - Memory);
      assert(Hash == Content->getHash() && "Hash not properly initialized");
      return Content;
    });
    // If we just generated it, return it!
    if (Generated)
      return PointerBase(cast<TrieContent>(Existing).getValuePointer());

    if (isa<TrieSubtrie>(Existing)) {
      S = &cast<TrieSubtrie>(Existing);
      Index = IndexGen.next();
      continue;
    }

    // Return the existing content if it's an exact match!
    auto &ExistingContent = cast<TrieContent>(Existing);
    if (ExistingContent.getHash() == Hash)
      return PointerBase(ExistingContent.getValuePointer());

    // Sink the existing content as long as the indexes match.
    for (;;) {
      size_t NextIndex = IndexGen.next();
      size_t NewIndexForExistingContent =
          IndexGen.getCollidingBits(ExistingContent.getHash());
      S = S->sink(Index, ExistingContent, IndexGen.getNumBits(),
                  NewIndexForExistingContent,
                  [&Impl](std::unique_ptr<TrieSubtrie> S) {
                    return Impl.save(std::move(S));
                  });
      Index = NextIndex;

      // Found the difference.
      if (NextIndex != NewIndexForExistingContent)
        break;
    }
  }
}

static void printHexDigit(raw_ostream &OS, uint8_t Digit) {
  if (Digit < 10)
    OS << char(Digit + '0');
  else
    OS << char(Digit - 10 + 'a');
}

static void printHexDigits(raw_ostream &OS, ArrayRef<uint8_t> Bytes,
                           size_t StartBit, size_t NumBits) {
  assert(StartBit % 4 == 0);
  assert(NumBits % 4 == 0);
  for (size_t I = StartBit, E = StartBit + NumBits; I != E; I += 4) {
    uint8_t HexPair = Bytes[I / 8];
    uint8_t HexDigit = I % 8 == 0 ? HexPair >> 4 : HexPair & 0xf;
    printHexDigit(OS, HexDigit);
  }
}

static void printBits(raw_ostream &OS, ArrayRef<uint8_t> Bytes, size_t StartBit,
                      size_t NumBits) {
  assert(StartBit + NumBits <= Bytes.size() * 8u);
  for (size_t I = StartBit, E = StartBit + NumBits; I != E; ++I) {
    uint8_t Byte = Bytes[I / 8];
    size_t ByteOffset = I % 8;
    if (size_t ByteShift = 8 - ByteOffset - 1)
      Byte >>= ByteShift;
    OS << (Byte & 0x1 ? '1' : '0');
  }
}

void TrieSubtrie::printHash(raw_ostream &OS, ArrayRef<uint8_t> Bytes) const {
  // afb[1c:00*01110*0]def
  size_t EndBit = StartBit + NumBits;
  size_t HashEndBit = Bytes.size() * 8u;

  size_t FirstBinaryBit = StartBit & ~0x3u;
  printHexDigits(OS, Bytes, 0, FirstBinaryBit);

  size_t LastBinaryBit = (EndBit + 3u) & ~0x3u;
  OS << "[";
  printBits(OS, Bytes, FirstBinaryBit, LastBinaryBit - FirstBinaryBit);
  OS << "]";

  printHexDigits(OS, Bytes, LastBinaryBit, HashEndBit - LastBinaryBit);
}

static void appendIndexBits(std::string &Prefix, size_t Index,
                            size_t NumSlots) {
  std::string Bits;
  for (size_t NumBits = 1u; NumBits < NumSlots; NumBits <<= 1) {
    Bits.push_back('0' + (Index & 0x1));
    Index >>= 1;
  }
  for (char Ch : llvm::reverse(Bits))
    Prefix += Ch;
}

static void printPrefix(raw_ostream &OS, StringRef Prefix) {
  while (Prefix.size() >= 4) {
    uint8_t Digit;
    bool ErrorParsingBinary = Prefix.take_front(4).getAsInteger(2, Digit);
    assert(!ErrorParsingBinary);
    (void)ErrorParsingBinary;
    printHexDigit(OS, Digit);
    Prefix = Prefix.drop_front(4);
  }
  if (!Prefix.empty())
    OS << "[" << Prefix << "]";
}

void TrieSubtrie::print(raw_ostream &OS, Optional<std::string> Prefix) const {
  if (!Prefix) {
    OS << "root";
    Prefix.emplace();
  } else {
    OS << "subtrie=";
    printPrefix(OS, *Prefix);
  }

  OS << " num-slots=" << Slots.size() << "\n";
  SmallVector<TrieSubtrie *> Subs;
  SmallVector<std::string> Prefixes;
  for (size_t I = 0, E = Slots.size(); I != E; ++I) {
    TrieNode *N = get(I);
    if (!N)
      continue;
    OS << "- index=" << I << " ";
    if (auto *S = dyn_cast<TrieSubtrie>(N)) {
      std::string SubtriePrefix = *Prefix;
      appendIndexBits(SubtriePrefix, I, Slots.size());
      OS << "subtrie=";
      printPrefix(OS, SubtriePrefix);
      OS << "\n";
      Subs.push_back(S);
      Prefixes.push_back(SubtriePrefix);
      continue;
    }
    auto *Content = cast<TrieContent>(N);
    OS << "content=";
    printHash(OS, Content->getHash());
    OS << "\n";
  }
  for (size_t I = 0, E = Subs.size(); I != E; ++I)
    Subs[I]->print(OS, Prefixes[I]);
}

void ThreadSafeHashMappedTrieBase::print(raw_ostream &OS) const {
  OS << "root-bits=" << NumRootBits << " subtrie-bits=" << NumSubtrieBits
     << "\n";
  if (ImplType *Impl = ImplPtr.load())
    Impl->Root.print(OS);
  else
    OS << "[no-root]\n";
}

LLVM_DUMP_METHOD void ThreadSafeHashMappedTrieBase::dump() const {
  print(dbgs());
}

ThreadSafeHashMappedTrieBase::ThreadSafeHashMappedTrieBase(
    size_t ContentAllocSize, size_t ContentAllocAlign, size_t ContentOffset,
    Optional<size_t> NumRootBits, Optional<size_t> NumSubtrieBits)
    : ContentAllocSize(ContentAllocSize), ContentAllocAlign(ContentAllocAlign),
      ContentOffset(ContentOffset),
      NumRootBits(NumRootBits ? *NumRootBits : DefaultNumRootBits),
      NumSubtrieBits(NumSubtrieBits ? *NumSubtrieBits : DefaultNumSubtrieBits),
      ImplPtr(nullptr) {
  assert((!NumRootBits || *NumRootBits < 20) &&
         "Root should have fewer than ~1M slots");
  assert((!NumSubtrieBits || *NumSubtrieBits < 10) &&
         "Subtries should have fewer than ~1K slots");
}

ThreadSafeHashMappedTrieBase::ThreadSafeHashMappedTrieBase(
    ThreadSafeHashMappedTrieBase &&RHS)
    : ContentAllocSize(RHS.ContentAllocSize),
      ContentAllocAlign(RHS.ContentAllocAlign),
      ContentOffset(RHS.ContentOffset), NumRootBits(RHS.NumRootBits),
      NumSubtrieBits(RHS.NumSubtrieBits) {
  // Steal the root from RHS.
  ImplPtr = RHS.ImplPtr.exchange(nullptr);
}

ThreadSafeHashMappedTrieBase::~ThreadSafeHashMappedTrieBase() {
  assert(!ImplPtr.load() && "Expected subclass to call destroyImpl()");
}

void ThreadSafeHashMappedTrieBase::destroyImpl(
    function_ref<void(void *)> Destructor) {
  std::unique_ptr<ImplType> Impl(ImplPtr.exchange(nullptr));
  if (!Impl)
    return;

  // Destroy content nodes throughout trie. Avoid destroying any subtries since
  // we need TrieNode::classof() to find the content nodes.
  //
  // FIXME: Once we have bitsets (see FIXME in TrieSubtrie class), use them
  // facilitate sparse iteration here.
  if (Destructor)
    for (TrieSubtrie *Trie = &Impl->Root; Trie; Trie = Trie->Next.load())
      for (auto &Slot : Trie->Slots)
        if (auto *Content = dyn_cast_or_null<TrieContent>(Slot.load()))
          Destructor(Content->getValuePointer());

  // Destroy the subtries. Incidentally, this destroys them in the reverse order
  // of saving.
  TrieSubtrie *Trie = Impl->Root.Next;
  while (Trie) {
    TrieSubtrie *Next = Trie->Next.exchange(nullptr);
    delete Trie;
    Trie = Next;
  }
}

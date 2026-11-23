//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/NamedValuesSchema.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace llvm::cas;

char NamedValuesSchema::ID = 0;
constexpr StringLiteral NamedValuesSchema::SchemaName;

void NamedValuesSchema::anchor() {}

bool NamedValuesSchema::isNode(const ObjectProxy &Node) const {
  // Load the first ref to check its content.
  if (Node.getNumReferences() < 1)
    return false;

  auto FirstRef = Node.getReference(0);
  return FirstRef == *NamedValuesKindRef;
}

NamedValuesSchema::NamedValuesSchema(cas::ObjectStore &CAS, Error &E)
    : NamedValuesSchema::RTTIExtends(CAS) {
  ErrorAsOutParameter EAOP(E);
  auto Kind = CAS.storeFromString({}, SchemaName);
  if (!Kind) {
    E = Kind.takeError();
    return;
  }
  NamedValuesKindRef = *Kind;
}

Expected<NamedValuesSchema> NamedValuesSchema::create(ObjectStore &CAS) {
  Error E = Error::success();
  NamedValuesSchema S(CAS, E);
  if (E)
    return std::move(E);
  return S;
}

size_t NamedValuesSchema::getNumEntries(NamedValuesProxy Values) const {
  return Values.getNumReferences() - 1;
}

Error NamedValuesSchema::forEachEntry(
    NamedValuesProxy Values,
    function_ref<Error(const NamedValuesEntry &)> Callback) const {
  for (size_t I = 0, IE = getNumEntries(Values); I != IE; ++I)
    if (Error E = Callback(loadEntry(Values, I)))
      return E;

  return Error::success();
}

NamedValuesEntry NamedValuesSchema::loadEntry(NamedValuesProxy Values,
                                              size_t I) const {
  StringRef Name = Values.getName(I);
  auto ObjectRef = Values.getReference(I + 1);

  return {Name, ObjectRef};
}

std::optional<size_t> NamedValuesSchema::lookupEntry(NamedValuesProxy Values,
                                                     StringRef Name) const {
  size_t NumNames = getNumEntries(Values);
  if (!NumNames)
    return std::nullopt;

  // Start with a binary search, if there are enough entries.
  // FIXME: MaxLinearSearchSize is a heuristic and not optimized.
  const size_t MaxLinearSearchSize = 4;
  size_t Last = NumNames;
  size_t First = 0;
  while (Last - First > MaxLinearSearchSize) {
    auto I = First + (Last - First) / 2;
    StringRef NameI = Values.getName(I);
    switch (Name.compare(NameI)) {
    case 0:
      return I;
    case -1:
      Last = I;
      break;
    case 1:
      First = I + 1;
      break;
    }
  }

  // Use a linear search for small list.
  for (; First != Last; ++First)
    if (Name == Values.getName(First))
      return First;

  return std::nullopt;
}

Expected<NamedValuesProxy> NamedValuesSchema::load(ObjectRef Object) const {
  auto Node = CAS.getProxy(Object);
  if (!Node)
    return Node.takeError();

  return load(*Node);
}

Expected<NamedValuesProxy> NamedValuesSchema::load(ObjectProxy Object) const {
  if (!isNode(Object))
    return createStringError(inconvertibleErrorCode(),
                             "object does not conform to NamedValuesSchema");

  return NamedValuesProxy(*this, Object);
}

Expected<NamedValuesProxy>
NamedValuesSchema::construct(ArrayRef<NamedValuesEntry> Entries) {
  // ScratchPad for output.
  SmallString<256> Data;
  SmallVector<ObjectRef, 16> Refs;
  Refs.push_back(*NamedValuesKindRef);

  // Ensure a stable order for entries and ignore name collisions.
  SmallVector<NamedValuesEntry> Sorted(Entries);
  llvm::stable_sort(Sorted);

  if (llvm::unique(Sorted) != Sorted.end())
    return createStringError("entry names are not unique");

  raw_svector_ostream OS(Data);
  support::endian::Writer Writer(OS, endianness::little);
  // Encode the entries in the Data. The layout of the named values schema
  // object is:
  // * Name offset table: The offset of in the data blob for where to find the
  //   string. It has N + 1 entries and you can find the name of n-th entry at
  //   offset[n] -> offset[n+1]. Each offset is encoded as little-endian
  //   uint32_t.
  // * Object: ObjectRef for each entry is at n + 1 refs for the object (with
  //   the first one being the named value kind ID).

  // Write Name.
  // The start of the string table index.
  uint32_t StrIdx = sizeof(uint32_t) * (Sorted.size() + 1);
  for (auto &Entry : Sorted) {
    Writer.write(StrIdx);
    StrIdx += Entry.Name.size();

    // Append refs.
    Refs.push_back(Entry.Ref);
  }
  // Write the end index for the last string.
  Writer.write(StrIdx);

  // Write names in the end of the block.
  for (auto &Entry : Sorted)
    OS << Entry.Name;

  auto Proxy = CAS.createProxy(Refs, Data);
  if (!Proxy)
    return Proxy.takeError();

  return NamedValuesProxy(*this, *Proxy);
}

void NamedValuesSchema::Builder::add(StringRef Name, ObjectRef Ref) {
  StringSaver Saver(Alloc);
  Nodes.emplace_back(Saver.save(Name), Ref);
}

Expected<NamedValuesProxy> NamedValuesSchema::Builder::build() {
  auto Schema = NamedValuesSchema::create(CAS);
  if (!Schema)
    return Schema.takeError();
  return Schema->construct(Nodes);
}

StringRef NamedValuesProxy::getName(size_t I) const {
  uint32_t StartIdx =
      support::endian::read32le(getData().data() + sizeof(uint32_t) * I);
  uint32_t EndIdx =
      support::endian::read32le(getData().data() + sizeof(uint32_t) * (I + 1));

  return StringRef(getData().data() + StartIdx, EndIdx - StartIdx);
}

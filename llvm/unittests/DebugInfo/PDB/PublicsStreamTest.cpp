//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/PublicsStream.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/GSIStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/PDBFileBuilder.h"
#include "llvm/DebugInfo/PDB/Native/SymbolStream.h"
#include "llvm/Support/BinaryByteStream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::pdb;

namespace {
struct PublicSym {
  llvm::StringRef Name;
  uint16_t Segment;
  uint32_t Offset;
};

class MockPublics {
public:
  MockPublics(size_t StreamSize, BumpPtrAllocator &Alloc,
              msf::MSFBuilder Builder);
  static Expected<std::unique_ptr<MockPublics>>
  create(BumpPtrAllocator &Allocator, size_t StreamSize);

  void addPublics(ArrayRef<PublicSym> Syms);
  Error finish();

  PublicsStream *publicsStream();
  SymbolStream *symbolStream();

  MutableBinaryByteStream &stream() { return Stream; }

private:
  MutableBinaryByteStream Stream;

  msf::MSFBuilder MsfBuilder;
  std::optional<msf::MSFLayout> MsfLayout;

  GSIStreamBuilder Gsi;

  std::unique_ptr<PublicsStream> Publics;
  std::unique_ptr<SymbolStream> Symbols;
};

MockPublics::MockPublics(size_t StreamSize, BumpPtrAllocator &Allocator,
                         msf::MSFBuilder Builder)
    : Stream({Allocator.Allocate<uint8_t>(StreamSize), StreamSize},
             llvm::endianness::little),
      MsfBuilder(std::move(Builder)), Gsi(this->MsfBuilder) {}

Expected<std::unique_ptr<MockPublics>>
MockPublics::create(BumpPtrAllocator &Allocator, size_t StreamSize) {
  auto ExpectedMsf = msf::MSFBuilder::create(Allocator, 4096);
  if (!ExpectedMsf)
    return ExpectedMsf.takeError();
  return std::make_unique<MockPublics>(StreamSize, Allocator,
                                       std::move(*ExpectedMsf));
}

void MockPublics::addPublics(ArrayRef<PublicSym> Publics) {
  std::vector<BulkPublic> Bulks;
  for (const auto &Sym : Publics) {
    BulkPublic BP;
    BP.Name = Sym.Name.data();
    BP.NameLen = Sym.Name.size();
    BP.Offset = Sym.Offset;
    BP.Segment = Sym.Segment;
    Bulks.emplace_back(BP);
  }
  Gsi.addPublicSymbols(std::move(Bulks));
}

Error MockPublics::finish() {
  auto Err = Gsi.finalizeMsfLayout();
  if (Err)
    return Err;

  auto ExpectedLayout = MsfBuilder.generateLayout();
  if (!ExpectedLayout)
    return ExpectedLayout.takeError();
  MsfLayout = std::move(*ExpectedLayout);

  return Gsi.commit(*MsfLayout, Stream);
}

PublicsStream *MockPublics::publicsStream() {
  if (!Publics) {
    Publics = std::make_unique<PublicsStream>(
        msf::MappedBlockStream::createIndexedStream(*MsfLayout, Stream,
                                                    Gsi.getPublicsStreamIndex(),
                                                    MsfBuilder.getAllocator()));
  }
  return Publics.get();
}

SymbolStream *MockPublics::symbolStream() {
  if (!Symbols) {
    Symbols = std::make_unique<SymbolStream>(
        msf::MappedBlockStream::createIndexedStream(*MsfLayout, Stream,
                                                    Gsi.getRecordStreamIndex(),
                                                    MsfBuilder.getAllocator()));
  }
  return Symbols.get();
}

std::array GSymbols{
    PublicSym{"??0Base@@QEAA@XZ", /*Segment=*/1, /*Offset=*/0},
    PublicSym{"??0Derived@@QEAA@XZ", /*Segment=*/1, /*Offset=*/32},
    PublicSym{"??0Derived2@@QEAA@XZ", /*Segment=*/1, /*Offset=*/32},
    PublicSym{"??0Derived3@@QEAA@XZ", /*Segment=*/1, /*Offset=*/80},
    PublicSym{"??1Base@@UEAA@XZ", /*Segment=*/1, /*Offset=*/160},
    PublicSym{"??1Derived@@UEAA@XZ", /*Segment=*/1, /*Offset=*/176},
    PublicSym{"??1Derived2@@UEAA@XZ", /*Segment=*/1, /*Offset=*/176},
    PublicSym{"??1Derived3@@UEAA@XZ", /*Segment=*/1, /*Offset=*/208},
    PublicSym{"??3@YAXPEAX_K@Z", /*Segment=*/1, /*Offset=*/256},
    PublicSym{"??_EDerived3@@W7EAAPEAXI@Z", /*Segment=*/1, /*Offset=*/268},
    PublicSym{"??_GBase@@UEAAPEAXI@Z", /*Segment=*/1, /*Offset=*/288},
    PublicSym{"??_EBase@@UEAAPEAXI@Z", /*Segment=*/1, /*Offset=*/288},
    PublicSym{"??_EDerived2@@UEAAPEAXI@Z", /*Segment=*/1, /*Offset=*/352},
    PublicSym{"??_EDerived@@UEAAPEAXI@Z", /*Segment=*/1, /*Offset=*/352},
    PublicSym{"??_GDerived@@UEAAPEAXI@Z", /*Segment=*/1, /*Offset=*/352},
    PublicSym{"??_GDerived2@@UEAAPEAXI@Z", /*Segment=*/1, /*Offset=*/352},
    PublicSym{"??_EDerived3@@UEAAPEAXI@Z", /*Segment=*/1, /*Offset=*/416},
    PublicSym{"??_GDerived3@@UEAAPEAXI@Z", /*Segment=*/1, /*Offset=*/416},
    PublicSym{"?AMethod@AClass@@QEAAXHPEAD@Z", /*Segment=*/1, /*Offset=*/480},
    PublicSym{"?Something@AClass@@SA_ND@Z", /*Segment=*/1, /*Offset=*/496},
    PublicSym{"?dup1@@YAHH@Z", /*Segment=*/1, /*Offset=*/544},
    PublicSym{"?dup3@@YAHH@Z", /*Segment=*/1, /*Offset=*/544},
    PublicSym{"?dup2@@YAHH@Z", /*Segment=*/1, /*Offset=*/544},
    PublicSym{"?foobar@@YAHH@Z", /*Segment=*/1, /*Offset=*/560},
    PublicSym{"main", /*Segment=*/1, /*Offset=*/576},
    PublicSym{"??_7Base@@6B@", /*Segment=*/2, /*Offset=*/0},
    PublicSym{"??_7Derived@@6B@", /*Segment=*/2, /*Offset=*/8},
    PublicSym{"??_7Derived2@@6B@", /*Segment=*/2, /*Offset=*/8},
    PublicSym{"??_7Derived3@@6BDerived2@@@", /*Segment=*/2, /*Offset=*/16},
    PublicSym{"??_7Derived3@@6BDerived@@@", /*Segment=*/2, /*Offset=*/24},
    PublicSym{"?AGlobal@@3HA", /*Segment=*/3, /*Offset=*/0},
};

} // namespace

static std::pair<uint32_t, uint32_t>
nthSymbolAddress(PublicsStream *Publics, SymbolStream *Symbols, size_t N) {
  auto Index = Publics->getAddressMap()[N].value();
  codeview::CVSymbol Sym = Symbols->readRecord(Index);
  auto ExpectedPub =
      codeview::SymbolDeserializer::deserializeAs<codeview::PublicSym32>(Sym);
  if (!ExpectedPub)
    return std::pair(0, 0);
  return std::pair(ExpectedPub->Segment, ExpectedPub->Offset);
}

TEST(PublicsStreamTest, FindByAddress) {
  BumpPtrAllocator Allocator;
  auto ExpectedMock = MockPublics::create(Allocator, 1 << 20);
  ASSERT_TRUE(bool(ExpectedMock));
  std::unique_ptr<MockPublics> Mock = std::move(*ExpectedMock);

  Mock->addPublics(GSymbols);
  Error Err = Mock->finish();
  ASSERT_FALSE(Err) << Err;

  auto *Publics = Mock->publicsStream();
  ASSERT_NE(Publics, nullptr);
  Err = Publics->reload();
  ASSERT_FALSE(Err) << Err;

  auto *Symbols = Mock->symbolStream();
  ASSERT_NE(Symbols, nullptr);
  Err = Symbols->reload();
  ASSERT_FALSE(Err) << Err;

  auto VTableDerived = Publics->findByAddress(*Symbols, 2, 8);
  ASSERT_TRUE(VTableDerived.has_value());
  // both derived and derived2 have their vftables there - but derived2 is first
  // (due to ICF)
  ASSERT_EQ(VTableDerived->first.Name, "??_7Derived2@@6B@");
  ASSERT_EQ(VTableDerived->second, 26u);

  // Again, make sure that we find the first symbol
  auto VectorDtorDerived = Publics->findByAddress(*Symbols, 1, 352);
  ASSERT_TRUE(VectorDtorDerived.has_value());
  ASSERT_EQ(VectorDtorDerived->first.Name, "??_EDerived2@@UEAAPEAXI@Z");
  ASSERT_EQ(VectorDtorDerived->second, 12u);
  ASSERT_EQ(nthSymbolAddress(Publics, Symbols, 13), std::pair(1u, 352u));
  ASSERT_EQ(nthSymbolAddress(Publics, Symbols, 14), std::pair(1u, 352u));
  ASSERT_EQ(nthSymbolAddress(Publics, Symbols, 15), std::pair(1u, 352u));
  ASSERT_EQ(nthSymbolAddress(Publics, Symbols, 16), std::pair(1u, 416u));

  ASSERT_FALSE(Publics->findByAddress(*Symbols, 2, 7).has_value());
  ASSERT_FALSE(Publics->findByAddress(*Symbols, 2, 9).has_value());

  auto GlobalSym = Publics->findByAddress(*Symbols, 3, 0);
  ASSERT_TRUE(GlobalSym.has_value());
  ASSERT_EQ(GlobalSym->first.Name, "?AGlobal@@3HA");
  ASSERT_EQ(GlobalSym->second, 30u);

  // test corrupt debug info
  codeview::CVSymbol GlobalCVSym =
      Symbols->readRecord(Publics->getAddressMap()[30]);
  ASSERT_EQ(GlobalCVSym.kind(), codeview::S_PUB32);
  // CVSymbol::data returns a pointer to const data, so we modify the backing
  // data
  uint8_t *PDBData = Mock->stream().data().data();
  auto Offset = GlobalCVSym.data().data() - PDBData;
  reinterpret_cast<codeview::RecordPrefix *>(PDBData + Offset)->RecordKind =
      codeview::S_GDATA32;
  ASSERT_EQ(GlobalCVSym.kind(), codeview::S_GDATA32);

  GlobalSym = Publics->findByAddress(*Symbols, 3, 0);
  ASSERT_FALSE(GlobalSym.has_value());
}

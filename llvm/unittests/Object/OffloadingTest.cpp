#include "llvm/Object/OffloadBinary.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <optional>
#include <random>

using namespace llvm;
using namespace llvm::object;

TEST(OffloadingTest, checkOffloadingBinary) {
  // Create random data to fill the image.
  std::mt19937 Rng(std::random_device{}());
  std::uniform_int_distribution<uint64_t> SizeDist(0, 256);
  std::uniform_int_distribution<uint16_t> KindDist(0);
  std::uniform_int_distribution<uint16_t> BinaryDist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
  std::uniform_int_distribution<int16_t> StringDist('!', '~');
  std::vector<uint8_t> Image(SizeDist(Rng));
  std::generate(Image.begin(), Image.end(), [&]() { return BinaryDist(Rng); });
  std::vector<std::pair<std::string, std::string>> Strings(SizeDist(Rng));
  for (auto &KeyAndValue : Strings) {
    std::string Key(SizeDist(Rng), '\0');
    std::string Value(SizeDist(Rng), '\0');

    std::generate(Key.begin(), Key.end(), [&]() { return StringDist(Rng); });
    std::generate(Value.begin(), Value.end(),
                  [&]() { return StringDist(Rng); });

    KeyAndValue = std::make_pair(Key, Value);
  }

  // Create the image.
  MapVector<StringRef, StringRef> StringData;
  for (auto &KeyAndValue : Strings)
    StringData[KeyAndValue.first] = KeyAndValue.second;
  std::unique_ptr<MemoryBuffer> ImageData = MemoryBuffer::getMemBuffer(
      {reinterpret_cast<char *>(Image.data()), Image.size()}, "", false);

  OffloadBinary::OffloadingImage Data;
  Data.TheImageKind = static_cast<ImageKind>(KindDist(Rng));
  Data.TheOffloadKind = static_cast<OffloadKind>(KindDist(Rng));
  Data.Flags = KindDist(Rng);
  Data.StringData = StringData;
  Data.Image = std::move(ImageData);

  auto BinaryBuffer =
      MemoryBuffer::getMemBufferCopy(OffloadBinary::write(Data));
  auto BinaryOrErr = OffloadBinary::create(*BinaryBuffer);
  if (!BinaryOrErr)
    FAIL();

  // Make sure we get the same data out.
  auto &Binaries = *BinaryOrErr;
  ASSERT_EQ(Binaries.size(), 1u);
  auto &Binary = *Binaries[0];
  ASSERT_EQ(Data.TheImageKind, Binary.getImageKind());
  ASSERT_EQ(Data.TheOffloadKind, Binary.getOffloadKind());
  ASSERT_EQ(Data.Flags, Binary.getFlags());

  for (auto &KeyAndValue : Strings)
    ASSERT_TRUE(StringData[KeyAndValue.first] ==
                Binary.getString(KeyAndValue.first));

  EXPECT_TRUE(Data.Image->getBuffer() == Binary.getImage());

  // Ensure the size and alignment of the data is correct.
  EXPECT_TRUE(Binary.getSize() % OffloadBinary::getAlignment() == 0);
  EXPECT_TRUE(Binary.getSize() == BinaryBuffer->getBuffer().size());
}

static std::unique_ptr<MemoryBuffer>
createMultiEntryBinary(size_t NumEntries,
                       SmallVectorImpl<std::string> &StringStorage) {
  // Reserve space to prevent reallocation which would invalidate StringRefs.
  // Each entry needs: "id", id_value, "arch", arch_value, image_content = 5
  // strings.
  StringStorage.reserve(NumEntries * 5);

  SmallVector<OffloadBinary::OffloadingImage> Images;

  for (size_t i = 0; i < NumEntries; ++i) {
    OffloadBinary::OffloadingImage Data;
    Data.TheImageKind = static_cast<ImageKind>(i % IMG_LAST);
    Data.TheOffloadKind = static_cast<OffloadKind>(i % OFK_LAST);

    MapVector<StringRef, StringRef> StringData;

    StringStorage.push_back("id");
    StringStorage.push_back(std::to_string(i));
    StringData[StringStorage[StringStorage.size() - 2]] =
        StringStorage[StringStorage.size() - 1];

    StringStorage.push_back("arch");
    StringStorage.push_back("gpu" + std::to_string(i));
    StringData[StringStorage[StringStorage.size() - 2]] =
        StringStorage[StringStorage.size() - 1];

    Data.StringData = StringData;

    // Make the last entry metadata-only (no image)
    if (i == NumEntries - 1) {
      Data.Flags = OIF_Metadata;
      Data.Image = MemoryBuffer::getMemBuffer("", "", false);
    } else {
      Data.Flags = i * 100;
      StringStorage.push_back("ImageData" + std::to_string(i));
      Data.Image = MemoryBuffer::getMemBuffer(StringStorage.back(), "", false);
    }

    Images.push_back(std::move(Data));
  }

  return MemoryBuffer::getMemBufferCopy(OffloadBinary::write(Images));
}

// Test multi-entry binaries and extraction without index (get all entries).
TEST(OffloadingTest, checkMultiEntryBinaryExtraction) {
  const size_t NumEntries = 5;
  SmallVector<std::string> StringStorage;
  auto BinaryBuffer = createMultiEntryBinary(NumEntries, StringStorage);

  // Test extracting all entries (no index).
  auto BinariesOrErr = OffloadBinary::create(*BinaryBuffer);
  ASSERT_THAT_EXPECTED(BinariesOrErr, Succeeded());

  auto &Binaries = *BinariesOrErr;
  ASSERT_EQ(Binaries.size(), NumEntries)
      << "Expected all entries when no index provided";

  // Verify each entry.
  for (size_t i = 0; i < NumEntries; ++i) {
    auto &Binary = *Binaries[i];
    EXPECT_EQ(Binary.getImageKind(), static_cast<ImageKind>(i % IMG_LAST));
    EXPECT_EQ(Binary.getOffloadKind(), static_cast<OffloadKind>(i % OFK_LAST));
    EXPECT_EQ(Binary.getIndex(), i);

    std::string ExpectedId = std::to_string(i);
    std::string ExpectedArch = "gpu" + std::to_string(i);
    EXPECT_EQ(Binary.getString("id"), ExpectedId);
    EXPECT_EQ(Binary.getString("arch"), ExpectedArch);

    // Last entry is metadata-only.
    if (i == NumEntries - 1) {
      EXPECT_EQ(Binary.getFlags(), OIF_Metadata);
      EXPECT_TRUE(Binary.getImage().empty());
    } else {
      EXPECT_EQ(Binary.getFlags(), i * 100);
      std::string ExpectedImage = "ImageData" + std::to_string(i);
      EXPECT_EQ(Binary.getImage(), ExpectedImage);
    }
  }

  // Ensure the size and alignment of the data is correct.
  EXPECT_TRUE(Binaries[0]->getSize() % OffloadBinary::getAlignment() == 0);
  EXPECT_TRUE(Binaries[0]->getSize() == BinaryBuffer->getBuffer().size());
}

// Test index-based extraction from multi-entry binary.
TEST(OffloadingTest, checkIndexBasedExtraction) {
  const size_t NumEntries = 5;
  SmallVector<std::string> StringStorage;
  auto BinaryBuffer = createMultiEntryBinary(NumEntries, StringStorage);

  // Test extracting specific indices.
  for (uint64_t i = 0; i < NumEntries; ++i) {
    auto BinariesOrErr = OffloadBinary::create(*BinaryBuffer, i);
    ASSERT_THAT_EXPECTED(BinariesOrErr, Succeeded());

    auto &Binaries = *BinariesOrErr;
    ASSERT_EQ(Binaries.size(), 1u) << "Expected single entry when using index";

    auto &Binary = *Binaries[0];
    EXPECT_EQ(Binary.getImageKind(), static_cast<ImageKind>(i % IMG_LAST));
    EXPECT_EQ(Binary.getOffloadKind(), static_cast<OffloadKind>(i % OFK_LAST));
    EXPECT_EQ(Binary.getIndex(), i);

    std::string ExpectedId = std::to_string(i);
    std::string ExpectedArch = "gpu" + std::to_string(i);
    EXPECT_EQ(Binary.getString("id"), ExpectedId);
    EXPECT_EQ(Binary.getString("arch"), ExpectedArch);

    // Last entry is metadata-only.
    if (i == NumEntries - 1) {
      EXPECT_EQ(Binary.getFlags(), OIF_Metadata);
      EXPECT_TRUE(Binary.getImage().empty());
    } else {
      EXPECT_EQ(Binary.getFlags(), i * 100);
      std::string ExpectedImage = "ImageData" + std::to_string(i);
      EXPECT_EQ(Binary.getImage(), ExpectedImage);
    }
  }

  // Test out-of-bounds index.
  auto OutOfBoundsOrErr = OffloadBinary::create(*BinaryBuffer, NumEntries + 10);
  EXPECT_THAT_EXPECTED(OutOfBoundsOrErr, Failed());
}

TEST(OffloadingTest, checkEdgeCases) {
  // Test with empty string data.
  {
    OffloadBinary::OffloadingImage Data;
    Data.TheImageKind = IMG_Object;
    Data.TheOffloadKind = OFK_OpenMP;
    Data.Flags = 0;
    Data.StringData = MapVector<StringRef, StringRef>(); // Empty

    std::string ImageContent = "TestImage";
    Data.Image = MemoryBuffer::getMemBuffer(ImageContent, "", false);

    auto BinaryBuffer =
        MemoryBuffer::getMemBufferCopy(OffloadBinary::write(Data));
    auto BinariesOrErr = OffloadBinary::create(*BinaryBuffer);
    ASSERT_THAT_EXPECTED(BinariesOrErr, Succeeded());

    auto &Binaries = *BinariesOrErr;
    ASSERT_EQ(Binaries.size(), 1u);
    EXPECT_TRUE(Binaries[0]->strings().empty());
    EXPECT_EQ(Binaries[0]->getImage(), ImageContent);
  }

  // Test with empty image data.
  {
    std::string Key = "test";
    std::string Value = "value";

    OffloadBinary::OffloadingImage Data;
    Data.TheImageKind = IMG_Object;
    Data.TheOffloadKind = OFK_SYCL;
    Data.Flags = 0;

    MapVector<StringRef, StringRef> StringData;
    StringData[Key] = Value;
    Data.StringData = StringData;

    Data.Image = MemoryBuffer::getMemBuffer("", "", false); // Empty image

    auto BinaryBuffer =
        MemoryBuffer::getMemBufferCopy(OffloadBinary::write(Data));
    auto BinariesOrErr = OffloadBinary::create(*BinaryBuffer);
    ASSERT_THAT_EXPECTED(BinariesOrErr, Succeeded());

    auto &Binaries = *BinariesOrErr;
    ASSERT_EQ(Binaries.size(), 1u);
    EXPECT_TRUE(Binaries[0]->getImage().empty());
    EXPECT_EQ(Binaries[0]->getString("test"), "value");
  }

  // Test with large string values.
  {
    std::string Key = "large_key";
    std::string LargeValue(4096, 'X'); // Large value
    std::string ImageContent = "Image";

    OffloadBinary::OffloadingImage Data;
    Data.TheImageKind = IMG_Bitcode;
    Data.TheOffloadKind = OFK_OpenMP;
    Data.Flags = 0;

    MapVector<StringRef, StringRef> StringData;
    StringData[Key] = LargeValue;
    Data.StringData = StringData;

    Data.Image = MemoryBuffer::getMemBuffer(ImageContent, "", false);

    auto BinaryBuffer =
        MemoryBuffer::getMemBufferCopy(OffloadBinary::write(Data));
    auto BinariesOrErr = OffloadBinary::create(*BinaryBuffer);
    ASSERT_THAT_EXPECTED(BinariesOrErr, Succeeded());

    auto &Binaries = *BinariesOrErr;
    ASSERT_EQ(Binaries.size(), 1u);
    EXPECT_EQ(Binaries[0]->getString("large_key"), LargeValue);
    EXPECT_EQ(Binaries[0]->getString("large_key").size(), 4096u);
  }
}

TEST(OffloadingTest, checkCompressedRoundTrip) {
  std::optional<compression::Params> Params;
  if (compression::zstd::isAvailable())
    Params = compression::Params(compression::Format::Zstd);
  else if (compression::zlib::isAvailable())
    Params = compression::Params(compression::Format::Zlib);
  else
    GTEST_SKIP() << "compression is unsupported";

  std::string ImageContent(4096, 'A');
  ImageContent += std::string(4096, 'B');

  OffloadBinary::OffloadingImage Data;
  Data.TheImageKind = IMG_Object;
  Data.TheOffloadKind = OFK_HIP;
  Data.Flags = 7;
  Data.StringData["triple"] = "amdgcn-amd-amdhsa";
  Data.StringData["arch"] = "gfx90a";
  Data.Image = MemoryBuffer::getMemBuffer(ImageContent, "", false);

  SmallString<0> Uncompressed = OffloadBinary::write(Data);
  Expected<SmallString<0>> CompressedOrErr =
      OffloadBinary::write(Data, *Params);
  ASSERT_THAT_EXPECTED(CompressedOrErr, Succeeded());
  SmallString<0> &Compressed = *CompressedOrErr;
  EXPECT_LT(Compressed.size(), Uncompressed.size());

  // Verify that the uncompressed and compressed versions are identical.
  auto HeaderOrErr = OffloadBinary::extractHeader(MemoryBufferRef(
      StringRef(Compressed.data(), Compressed.size()), "compressed"));
  ASSERT_THAT_EXPECTED(HeaderOrErr, Succeeded());
  EXPECT_EQ(Compressed.size(),
            alignTo((*HeaderOrErr)->Size, OffloadBinary::getAlignment()));
  EXPECT_EQ((*HeaderOrErr)->InflatedSize, Uncompressed.size());
  StringRef Payload = StringRef(Compressed.data(), (*HeaderOrErr)->Size)
                          .drop_front((*HeaderOrErr)->EntriesOffset);
  EXPECT_EQ(identify_magic(Payload), Params->format == compression::Format::Zstd
                                         ? file_magic::zstd
                                         : file_magic::zlib);

  auto BinaryBuffer = MemoryBuffer::getMemBufferCopy(Compressed);
  auto BinariesOrErr = OffloadBinary::create(*BinaryBuffer);
  ASSERT_THAT_EXPECTED(BinariesOrErr, Succeeded());
  ASSERT_EQ(BinariesOrErr->size(), 1u);

  OffloadBinary &Binary = *(*BinariesOrErr)[0];
  EXPECT_EQ(Binary.getImageKind(), IMG_Object);
  EXPECT_EQ(Binary.getOffloadKind(), OFK_HIP);
  EXPECT_EQ(Binary.getFlags(), 7u);
  EXPECT_EQ(Binary.getTriple(), "amdgcn-amd-amdhsa");
  EXPECT_EQ(Binary.getArch(), "gfx90a");
  EXPECT_EQ(Binary.getImage(), ImageContent);
  EXPECT_EQ(Binary.getSize(), Uncompressed.size());

  // The concatenated form should still extract properly when compressed.
  SmallString<0> Concat = Compressed;
  Concat.append(Uncompressed);
  SmallVector<OffloadFile> Files;
  ASSERT_THAT_ERROR(
      extractOffloadBinaries(
          MemoryBufferRef(StringRef(Concat.data(), Concat.size()), "concat"),
          Files),
      Succeeded());
  ASSERT_EQ(Files.size(), 2u);
  EXPECT_EQ(Files[0].getBinary()->getImage(), ImageContent);
  EXPECT_EQ(Files[1].getBinary()->getImage(), ImageContent);
}

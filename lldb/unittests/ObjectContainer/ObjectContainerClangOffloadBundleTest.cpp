//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectContainer/Clang-Offload-Bundle/ObjectContainerClangOffloadBundle.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/OffloadBundle.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

using namespace lldb;
using namespace lldb_private;

namespace {

constexpr llvm::StringLiteral BundleMagic = "__CLANG_OFFLOAD_BUNDLE__";
constexpr llvm::StringLiteral BundleID = "hipv4-amdgcn-amd-amdhsa--gfx942";

void AppendU64(std::vector<uint8_t> &bytes, uint64_t value) {
  for (unsigned i = 0; i != 8; ++i)
    bytes.push_back(static_cast<uint8_t>(value >> (i * 8)));
}

std::vector<uint8_t> MakeBundle(llvm::ArrayRef<uint8_t> payload) {
  std::vector<uint8_t> bytes(BundleMagic.bytes_begin(),
                             BundleMagic.bytes_end());
  AppendU64(bytes, 1);
  const uint64_t payload_offset =
      BundleMagic.size() + 4 * sizeof(uint64_t) + BundleID.size();
  AppendU64(bytes, payload_offset);
  AppendU64(bytes, payload.size());
  AppendU64(bytes, BundleID.size());
  bytes.insert(bytes.end(), BundleID.bytes_begin(), BundleID.bytes_end());
  bytes.insert(bytes.end(), payload.begin(), payload.end());
  return bytes;
}

llvm::Expected<TestFile> MakeGPUELF() {
  return TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  OSABI:           ELFOSABI_AMDGPU_HSA
  ABIVersion:      0x1
  Type:            ET_DYN
  Machine:         EM_AMDGPU
  Flags:           [ EF_AMDGPU_MACH_AMDGCN_GFX942 ]
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x2000
    AddressAlign:    0x4
    Content:         '00000000'
...
)");
}

llvm::Expected<TestFile>
MakeELFContainingBundle(llvm::ArrayRef<uint8_t> bundle) {
  std::string yaml = "--- !ELF\n"
                     "FileHeader:\n"
                     "  Class:           ELFCLASS64\n"
                     "  Data:            ELFDATA2LSB\n"
                     "  Type:            ET_DYN\n"
                     "  Machine:         EM_X86_64\n"
                     "Sections:\n"
                     "  - Name:            .hip_fatbin\n"
                     "    Type:            SHT_PROGBITS\n"
                     "    Offset:          0x1000\n"
                     "    AddressAlign:    0x10\n"
                     "    Content:         " +
                     llvm::toHex(bundle) + "\n...\n";
  return TestFile::fromYaml(yaml);
}

class ObjectContainerClangOffloadBundleTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF,
                ObjectContainerClangOffloadBundle>
      subsystems;
};

} // namespace

TEST_F(ObjectContainerClangOffloadBundleTest, LoadsCompressedDeviceImage) {
  if (!llvm::compression::zlib::isAvailable())
    GTEST_SKIP() << "zlib is unavailable";

  auto gpu_file = MakeGPUELF();
  ASSERT_THAT_EXPECTED(gpu_file, llvm::Succeeded());
  DataExtractorSP gpu_data = gpu_file->moduleSpec().GetExtractor();
  llvm::ArrayRef<uint8_t> gpu_bytes(gpu_data->GetDataStart(),
                                    gpu_data->GetByteSize());

  std::vector<uint8_t> bundle = MakeBundle(gpu_bytes);
  auto bundle_buffer = llvm::MemoryBuffer::getMemBufferCopy(llvm::StringRef(
      reinterpret_cast<const char *>(bundle.data()), bundle.size()));
  auto compressed = llvm::object::CompressedOffloadBundle::compress(
      llvm::compression::Params(llvm::compression::Format::Zlib),
      *bundle_buffer, llvm::object::CompressedOffloadBundle::DefaultVersion);
  ASSERT_THAT_EXPECTED(compressed, llvm::Succeeded());

  llvm::StringRef compressed_bytes = (*compressed)->getBuffer();
  auto bundled_file = MakeELFContainingBundle(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(compressed_bytes.data()),
      compressed_bytes.size()));
  ASSERT_THAT_EXPECTED(bundled_file, llvm::Succeeded());

  llvm::Expected<llvm::sys::fs::TempFile> temp_file =
      bundled_file->writeToTemporaryFile();
  ASSERT_THAT_EXPECTED(temp_file, llvm::Succeeded());
  const std::string path = temp_file->TmpName;
  llvm::FileRemover file_remover(path);
  ASSERT_THAT_ERROR(temp_file->keep(), llvm::Succeeded());
  FileSpec file(path);

  ModuleSpecList specs = ObjectFile::GetModuleSpecifications(file, 0, 0);
  ASSERT_EQ(2u, specs.GetSize());

  ModuleSpec device_spec;
  ASSERT_TRUE(specs.GetModuleSpecAtIndex(1, device_spec));
  EXPECT_EQ(device_spec.GetObjectOffset(), 0u);
  EXPECT_EQ(device_spec.GetObjectSize(), gpu_bytes.size());
  ASSERT_NE(device_spec.GetExtractor(), nullptr);
  EXPECT_EQ(device_spec.GetArchitecture().GetClangTargetCPU(), "gfx942");

  auto module_sp = std::make_shared<Module>(device_spec);
  ASSERT_NE(module_sp->GetObjectFile(), nullptr);
  EXPECT_EQ(module_sp->GetObjectFile()->GetArchitecture().GetClangTargetCPU(),
            "gfx942");

  ModuleSpec requested(file);
  requested.GetArchitecture() = device_spec.GetArchitecture();
  auto selected_module_sp = std::make_shared<Module>(requested);
  ASSERT_NE(selected_module_sp->GetObjectFile(), nullptr);
  EXPECT_EQ(selected_module_sp->GetObjectFile()
                ->GetArchitecture()
                .GetClangTargetCPU(),
            "gfx942");
}

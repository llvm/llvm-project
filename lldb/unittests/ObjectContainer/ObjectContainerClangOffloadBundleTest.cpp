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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace lldb;
using namespace lldb_private;

namespace {

constexpr uint64_t BundleSectionOffset = 0x1000;
constexpr llvm::StringLiteral BundleMagic = "__CLANG_OFFLOAD_BUNDLE__";

void AppendU64(std::vector<uint8_t> &bytes, uint64_t value) {
  for (unsigned i = 0; i != 8; ++i)
    bytes.push_back(static_cast<uint8_t>(value >> (i * 8)));
}

std::vector<uint8_t> MakeBundle(llvm::StringRef id,
                                llvm::ArrayRef<uint8_t> payload,
                                std::optional<uint64_t> entry_offset = {}) {
  std::vector<uint8_t> bytes(BundleMagic.bytes_begin(),
                             BundleMagic.bytes_end());
  AppendU64(bytes, 1);
  const uint64_t payload_offset = entry_offset.value_or(
      BundleMagic.size() + 4 * sizeof(uint64_t) + id.size());
  AppendU64(bytes, payload_offset);
  AppendU64(bytes, payload.size());
  AppendU64(bytes, id.size());
  bytes.insert(bytes.end(), id.bytes_begin(), id.bytes_end());
  if (!entry_offset)
    bytes.insert(bytes.end(), payload.begin(), payload.end());
  return bytes;
}

struct BundleInput {
  llvm::StringRef id;
  llvm::ArrayRef<uint8_t> payload;
};

struct BundleData {
  std::vector<uint8_t> bytes;
  std::vector<uint64_t> entry_offsets;
};

BundleData MakeBundle(llvm::ArrayRef<BundleInput> inputs) {
  BundleData result;
  result.bytes.insert(result.bytes.end(), BundleMagic.bytes_begin(),
                      BundleMagic.bytes_end());
  AppendU64(result.bytes, inputs.size());

  uint64_t payload_offset = BundleMagic.size() + sizeof(uint64_t);
  for (const BundleInput &input : inputs)
    payload_offset += 3 * sizeof(uint64_t) + input.id.size();

  for (const BundleInput &input : inputs) {
    result.entry_offsets.push_back(payload_offset);
    AppendU64(result.bytes, payload_offset);
    AppendU64(result.bytes, input.payload.size());
    AppendU64(result.bytes, input.id.size());
    result.bytes.insert(result.bytes.end(), input.id.bytes_begin(),
                        input.id.bytes_end());
    payload_offset += input.payload.size();
  }

  for (const BundleInput &input : inputs)
    result.bytes.insert(result.bytes.end(), input.payload.begin(),
                        input.payload.end());
  return result;
}

llvm::Expected<TestFile> MakeGPUELF(llvm::StringRef elf_flag) {
  std::string yaml = R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  OSABI:           ELFOSABI_AMDGPU_HSA
  ABIVersion:      0x1
  Type:            ET_DYN
  Machine:         EM_AMDGPU
  Flags:           [ )" +
                     elf_flag.str() + R"( ]
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x2000
    AddressAlign:    0x4
    Content:         '00000000'
...
)";
  return TestFile::fromYaml(yaml);
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

llvm::Expected<TestFile>
MakeELFWithBundle(llvm::StringRef id, uint64_t &entry_offset,
                  uint64_t &entry_size,
                  std::optional<uint64_t> invalid_entry_offset = {}) {
  auto inner_file = MakeGPUELF("EF_AMDGPU_MACH_AMDGCN_GFX942");
  if (!inner_file)
    return inner_file.takeError();

  DataExtractorSP inner_data = inner_file->moduleSpec().GetExtractor();
  llvm::ArrayRef<uint8_t> inner_bytes(inner_data->GetDataStart(),
                                      inner_data->GetByteSize());
  std::vector<uint8_t> bundle =
      MakeBundle(id, inner_bytes, invalid_entry_offset);

  entry_offset = BundleSectionOffset + BundleMagic.size() +
                 4 * sizeof(uint64_t) + id.size();
  entry_size = inner_bytes.size();
  return MakeELFContainingBundle(bundle);
}

class ObjectContainerClangOffloadBundleTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF,
                ObjectContainerClangOffloadBundle>
      subsystems;
};

} // namespace

TEST_F(ObjectContainerClangOffloadBundleTest, FindsAndLoadsDeviceImage) {
  constexpr llvm::StringLiteral BundleID =
      "hipv4-amdgcn-amd-amdhsa--gfx942:xnack+";
  uint64_t entry_offset;
  uint64_t entry_size;
  auto bundled_file = MakeELFWithBundle(BundleID, entry_offset, entry_size);
  ASSERT_THAT_EXPECTED(bundled_file, llvm::Succeeded());

  llvm::Expected<llvm::sys::fs::TempFile> temp_file =
      bundled_file->writeToTemporaryFile();
  ASSERT_THAT_EXPECTED(temp_file, llvm::Succeeded());
  const std::string path = temp_file->TmpName;
  llvm::FileRemover file_remover(path);
  ASSERT_THAT_ERROR(temp_file->keep(), llvm::Succeeded());
  FileSpec file(path);

  ModuleSpec input_spec = bundled_file->moduleSpec();
  DataExtractorSP data = input_spec.GetExtractor();

  ModuleSpecList device_specs =
      ObjectContainerClangOffloadBundle::GetModuleSpecifications(
          file, data, /*file_offset=*/0, data->GetByteSize());
  ASSERT_EQ(device_specs.GetSize(), 1u);

  ModuleSpec device_spec;
  ASSERT_TRUE(device_specs.GetModuleSpecAtIndex(0, device_spec));
  EXPECT_EQ(device_spec.GetObjectOffset(), entry_offset);
  EXPECT_EQ(device_spec.GetObjectSize(), entry_size);
  EXPECT_EQ(device_spec.GetArchitecture().GetClangTargetCPU(), "gfx942");

  ModuleSpecList all_specs =
      ObjectFile::GetModuleSpecifications(file, /*file_offset=*/0,
                                          /*file_size=*/0);
  ASSERT_EQ(all_specs.GetSize(), 2u);

  ModuleSpec requested(file, device_spec.GetArchitecture());
  auto module_sp = std::make_shared<Module>(requested);
  EXPECT_EQ(module_sp->GetObjectOffset(), entry_offset);
  ASSERT_NE(module_sp->GetObjectFile(), nullptr);
  EXPECT_EQ(module_sp->GetObjectFile()->GetArchitecture().GetClangTargetCPU(),
            "gfx942");

  auto container_module_sp =
      std::make_shared<Module>(file, device_spec.GetArchitecture());
  ASSERT_NE(container_module_sp->GetObjectFile(), nullptr);
  EXPECT_EQ(container_module_sp->GetObjectFile()->GetFileOffset(),
            entry_offset);

  constexpr lldb::offset_t containing_object_offset = 0x80;
  ModuleSpecList nested_specs =
      ObjectContainerClangOffloadBundle::GetModuleSpecifications(
          FileSpec(), data, containing_object_offset, data->GetByteSize());
  ASSERT_EQ(nested_specs.GetSize(), 1u);
  ModuleSpec nested_spec;
  ASSERT_TRUE(nested_specs.GetModuleSpecAtIndex(0, nested_spec));
  EXPECT_EQ(nested_spec.GetObjectOffset(),
            containing_object_offset + entry_offset);
}

TEST_F(ObjectContainerClangOffloadBundleTest,
       FindsMultipleDeviceImagesAndSkipsEmptyHostImage) {
  auto gfx908_file = MakeGPUELF("EF_AMDGPU_MACH_AMDGCN_GFX908");
  ASSERT_THAT_EXPECTED(gfx908_file, llvm::Succeeded());
  auto gfx942_file = MakeGPUELF("EF_AMDGPU_MACH_AMDGCN_GFX942");
  ASSERT_THAT_EXPECTED(gfx942_file, llvm::Succeeded());

  DataExtractorSP gfx908_data = gfx908_file->moduleSpec().GetExtractor();
  DataExtractorSP gfx942_data = gfx942_file->moduleSpec().GetExtractor();
  std::array<BundleInput, 3> inputs = {{
      {"host-x86_64-unknown-linux--", {}},
      {"hipv4-amdgcn-amd-amdhsa--gfx908",
       {gfx908_data->GetDataStart(), gfx908_data->GetByteSize()}},
      {"hipv4-amdgcn-amd-amdhsa--gfx942",
       {gfx942_data->GetDataStart(), gfx942_data->GetByteSize()}},
  }};
  BundleData bundle = MakeBundle(inputs);
  auto bundled_file = MakeELFContainingBundle(bundle.bytes);
  ASSERT_THAT_EXPECTED(bundled_file, llvm::Succeeded());

  auto temp_file = bundled_file->writeToTemporaryFile();
  ASSERT_THAT_EXPECTED(temp_file, llvm::Succeeded());
  const std::string path = temp_file->TmpName;
  llvm::FileRemover file_remover(path);
  ASSERT_THAT_ERROR(temp_file->keep(), llvm::Succeeded());
  FileSpec file(path);

  ModuleSpec input_spec = bundled_file->moduleSpec();
  DataExtractorSP data = input_spec.GetExtractor();
  ModuleSpecList device_specs =
      ObjectContainerClangOffloadBundle::GetModuleSpecifications(
          file, data, /*file_offset=*/0, data->GetByteSize());
  ASSERT_EQ(device_specs.GetSize(), 2u);

  ModuleSpec gfx908_spec;
  ModuleSpec gfx942_spec;
  ASSERT_TRUE(device_specs.GetModuleSpecAtIndex(0, gfx908_spec));
  ASSERT_TRUE(device_specs.GetModuleSpecAtIndex(1, gfx942_spec));
  EXPECT_EQ(gfx908_spec.GetArchitecture().GetClangTargetCPU(), "gfx908");
  EXPECT_EQ(gfx942_spec.GetArchitecture().GetClangTargetCPU(), "gfx942");
  EXPECT_EQ(gfx908_spec.GetObjectOffset(),
            BundleSectionOffset + bundle.entry_offsets[1]);
  EXPECT_EQ(gfx942_spec.GetObjectOffset(),
            BundleSectionOffset + bundle.entry_offsets[2]);
  EXPECT_EQ(gfx908_spec.GetObjectSize(), gfx908_data->GetByteSize());
  EXPECT_EQ(gfx942_spec.GetObjectSize(), gfx942_data->GetByteSize());

  ModuleSpecList all_specs =
      ObjectFile::GetModuleSpecifications(file, /*file_offset=*/0,
                                          /*file_size=*/0);
  EXPECT_EQ(all_specs.GetSize(), 3u);

  ModuleSpec requested(file, gfx942_spec.GetArchitecture());
  auto module_sp = std::make_shared<Module>(requested);
  ASSERT_NE(module_sp->GetObjectFile(), nullptr);
  EXPECT_EQ(module_sp->GetObjectFile()->GetArchitecture().GetClangTargetCPU(),
            "gfx942");
  EXPECT_EQ(module_sp->GetObjectFile()->GetFileOffset(),
            gfx942_spec.GetObjectOffset());
}

TEST_F(ObjectContainerClangOffloadBundleTest, IgnoresOutOfBoundsEntry) {
  uint64_t entry_offset;
  uint64_t entry_size;
  auto bundled_file =
      MakeELFWithBundle("hipv4-amdgcn-amd-amdhsa--gfx942", entry_offset,
                        entry_size, uint64_t{1} << 40);
  ASSERT_THAT_EXPECTED(bundled_file, llvm::Succeeded());

  ModuleSpec spec = bundled_file->moduleSpec();
  DataExtractorSP data = spec.GetExtractor();
  ModuleSpecList device_specs =
      ObjectContainerClangOffloadBundle::GetModuleSpecifications(
          FileSpec(), data, /*file_offset=*/0, data->GetByteSize());
  EXPECT_EQ(device_specs.GetSize(), 0u);
}

TEST_F(ObjectContainerClangOffloadBundleTest,
       RejectsOverflowingContainingOffset) {
  uint64_t entry_offset;
  uint64_t entry_size;
  auto bundled_file = MakeELFWithBundle("hipv4-amdgcn-amd-amdhsa--gfx942",
                                        entry_offset, entry_size);
  ASSERT_THAT_EXPECTED(bundled_file, llvm::Succeeded());

  ModuleSpec spec = bundled_file->moduleSpec();
  DataExtractorSP data = spec.GetExtractor();
  ModuleSpecList start_overflow_specs =
      ObjectContainerClangOffloadBundle::GetModuleSpecifications(
          FileSpec(), data,
          std::numeric_limits<uint64_t>::max() - entry_offset + 1,
          data->GetByteSize());
  EXPECT_EQ(start_overflow_specs.GetSize(), 0u);

  ModuleSpecList end_overflow_specs =
      ObjectContainerClangOffloadBundle::GetModuleSpecifications(
          FileSpec(), data, std::numeric_limits<uint64_t>::max() - entry_offset,
          data->GetByteSize());
  EXPECT_EQ(end_overflow_specs.GetSize(), 0u);
}

TEST_F(ObjectContainerClangOffloadBundleTest,
       RecognizesProcessorQualifiedArchitecture) {
  uint64_t entry_offset;
  uint64_t entry_size;
  auto bundled_file = MakeELFWithBundle("hipv4-amdgpu9.42-amd-amdhsa--",
                                        entry_offset, entry_size);
  ASSERT_THAT_EXPECTED(bundled_file, llvm::Succeeded());

  ModuleSpec spec = bundled_file->moduleSpec();
  DataExtractorSP data = spec.GetExtractor();
  ModuleSpecList device_specs =
      ObjectContainerClangOffloadBundle::GetModuleSpecifications(
          FileSpec(), data, /*file_offset=*/0, data->GetByteSize());
  ASSERT_EQ(device_specs.GetSize(), 1u);

  ModuleSpec device_spec;
  ASSERT_TRUE(device_specs.GetModuleSpecAtIndex(0, device_spec));
  EXPECT_EQ(device_spec.GetArchitecture().GetClangTargetCPU(), "gfx942");
}

TEST_F(ObjectContainerClangOffloadBundleTest,
       RecognizesGenericProcessorArchitecture) {
  uint64_t entry_offset;
  uint64_t entry_size;
  auto bundled_file = MakeELFWithBundle("hipv4-amdgpu9.4-amd-amdhsa--",
                                        entry_offset, entry_size);
  ASSERT_THAT_EXPECTED(bundled_file, llvm::Succeeded());

  ModuleSpec spec = bundled_file->moduleSpec();
  DataExtractorSP data = spec.GetExtractor();
  ModuleSpecList device_specs =
      ObjectContainerClangOffloadBundle::GetModuleSpecifications(
          FileSpec(), data, /*file_offset=*/0, data->GetByteSize());
  ASSERT_EQ(device_specs.GetSize(), 1u);

  ModuleSpec device_spec;
  ASSERT_TRUE(device_specs.GetModuleSpecAtIndex(0, device_spec));
  EXPECT_EQ(device_spec.GetArchitecture().GetClangTargetCPU(),
            "gfx9-4-generic");
}

TEST_F(ObjectContainerClangOffloadBundleTest, RejectsInvalidArchitecture) {
  uint64_t entry_offset;
  uint64_t entry_size;
  auto bundled_file = MakeELFWithBundle("hipv4-amdgpufoo-amd-amdhsa--gfx942",
                                        entry_offset, entry_size);
  ASSERT_THAT_EXPECTED(bundled_file, llvm::Succeeded());

  ModuleSpec spec = bundled_file->moduleSpec();
  DataExtractorSP data = spec.GetExtractor();
  ModuleSpecList device_specs =
      ObjectContainerClangOffloadBundle::GetModuleSpecifications(
          FileSpec(), data, /*file_offset=*/0, data->GetByteSize());
  EXPECT_EQ(device_specs.GetSize(), 0u);
}

TEST_F(ObjectContainerClangOffloadBundleTest, PlainELFHasOneSpecification) {
  auto plain_file = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
...
)");
  ASSERT_THAT_EXPECTED(plain_file, llvm::Succeeded());

  ModuleSpec spec = plain_file->moduleSpec();
  DataExtractorSP data = spec.GetExtractor();
  ModuleSpecList specs = ObjectFile::GetModuleSpecifications(
      FileSpec(), data, /*file_offset=*/0, data->GetByteSize());
  EXPECT_EQ(specs.GetSize(), 1u);
}

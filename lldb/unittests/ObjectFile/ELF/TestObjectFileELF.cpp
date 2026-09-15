//===-- TestObjectFileELF.cpp ---------------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

class ObjectFileELFTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF, SymbolFileSymtab>
      subsystems;
};

TEST_F(ObjectFileELFTest, SectionsResolveConsistently) {
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
  Entry:           0x0000000000400180
Sections:
  - Name:            .note.gnu.build-id
    Type:            SHT_NOTE
    Flags:           [ SHF_ALLOC ]
    Address:         0x0000000000400158
    AddressAlign:    0x0000000000000004
    Content:         040000001400000003000000474E55003F3EC29E3FD83E49D18C4D49CD8A730CC13117B6
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000400180
    AddressAlign:    0x0000000000000010
    Content:         554889E58B042500106000890425041060005DC3
  - Name:            .data
    Type:            SHT_PROGBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x0000000000601000
    AddressAlign:    0x0000000000000004
    Content:         2F000000
  - Name:            .bss
    Type:            SHT_NOBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x0000000000601004
    AddressAlign:    0x0000000000000004
    Size:            0x0000000000000004
Symbols:
  - Name:            Y
    Type:            STT_OBJECT
    Section:         .data
    Value:           0x0000000000601000
    Size:            0x0000000000000004
    Binding:         STB_GLOBAL
  - Name:            _start
    Type:            STT_FUNC
    Section:         .text
    Value:           0x0000000000400180
    Size:            0x0000000000000014
    Binding:         STB_GLOBAL
  - Name:            X
    Type:            STT_OBJECT
    Section:         .bss
    Value:           0x0000000000601004
    Size:            0x0000000000000004
    Binding:         STB_GLOBAL
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  SectionList *list = module_sp->GetSectionList();
  ASSERT_NE(nullptr, list);

  auto bss_sp = list->FindSectionByName(".bss");
  ASSERT_NE(nullptr, bss_sp);
  auto data_sp = list->FindSectionByName(".data");
  ASSERT_NE(nullptr, data_sp);
  auto text_sp = list->FindSectionByName(".text");
  ASSERT_NE(nullptr, text_sp);

  const Symbol *X = module_sp->FindFirstSymbolWithNameAndType(ConstString("X"),
                                                              eSymbolTypeAny);
  ASSERT_NE(nullptr, X);
  EXPECT_EQ(bss_sp, X->GetAddress().GetSection());

  const Symbol *Y = module_sp->FindFirstSymbolWithNameAndType(ConstString("Y"),
                                                              eSymbolTypeAny);
  ASSERT_NE(nullptr, Y);
  EXPECT_EQ(data_sp, Y->GetAddress().GetSection());

  const Symbol *start = module_sp->FindFirstSymbolWithNameAndType(
      ConstString("_start"), eSymbolTypeAny);
  ASSERT_NE(nullptr, start);
  EXPECT_EQ(text_sp, start->GetAddress().GetSection());
}

// Test that GetModuleSpecifications works on an "atypical" object file which
// has section headers right after the ELF header (instead of the more common
// layout where the section headers are at the very end of the object file).
//
// Test file generated with yaml2obj (@svn rev 324254) from the following input:
/*
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
  Entry:           0x00000000004003D0
Sections:
  - Name:            .note.gnu.build-id
    Type:            SHT_NOTE
    Flags:           [ SHF_ALLOC ]
    Address:         0x0000000000400274
    AddressAlign:    0x0000000000000004
    Content:         040000001400000003000000474E55001B8A73AC238390E32A7FF4AC8EBE4D6A41ECF5C9
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x00000000004003D0
    AddressAlign:    0x0000000000000010
    Content:         DEADBEEFBAADF00D
...
*/
TEST_F(ObjectFileELFTest, GetModuleSpecifications_EarlySectionHeaders) {
  std::string SO = GetInputFilePath("early-section-headers.so");
  ModuleSpecList Specs =
      ObjectFile::GetModuleSpecifications(FileSpec(SO), 0, 0);
  ASSERT_EQ(Specs.GetSize(), 1u);
  ModuleSpec Spec;
  ASSERT_TRUE(Specs.GetModuleSpecAtIndex(0, Spec)) ;
  UUID Uuid;
  Uuid.SetFromStringRef("1b8a73ac238390e32a7ff4ac8ebe4d6a41ecf5c9");
  EXPECT_EQ(Spec.GetUUID(), Uuid);
}

TEST_F(ObjectFileELFTest, GetModuleSpecifications_OffsetSizeWithNormalFile) {
  std::string SO = GetInputFilePath("liboffset-test.so");
  ModuleSpecList Specs =
      ObjectFile::GetModuleSpecifications(FileSpec(SO), 0, 0);
  ASSERT_EQ(Specs.GetSize(), 1u);
  ModuleSpec Spec;
  ASSERT_TRUE(Specs.GetModuleSpecAtIndex(0, Spec)) ;
  UUID Uuid;
  Uuid.SetFromStringRef("7D6E4738");
  EXPECT_EQ(Spec.GetUUID(), Uuid);
  EXPECT_EQ(Spec.GetObjectOffset(), 0UL);
  EXPECT_EQ(Spec.GetObjectSize(), 3600UL);
  EXPECT_EQ(FileSystem::Instance().GetByteSize(FileSpec(SO)), 3600UL);
}

TEST_F(ObjectFileELFTest, GetModuleSpecifications_OffsetSizeWithOffsetFile) {
  // The contents of offset-test.bin are
  // -    0-1023: \0
  // - 1024-4623: liboffset-test.so (offset: 1024, size: 3600, CRC32: 7D6E4738)
  // - 4624-4639: \0
  std::string SO = GetInputFilePath("offset-test.bin");
  ModuleSpecList Specs =
      ObjectFile::GetModuleSpecifications(FileSpec(SO), 1024, 3600);
  ASSERT_EQ(Specs.GetSize(), 1u);
  ModuleSpec Spec;
  ASSERT_TRUE(Specs.GetModuleSpecAtIndex(0, Spec)) ;
  UUID Uuid;
  Uuid.SetFromStringRef("7D6E4738");
  EXPECT_EQ(Spec.GetUUID(), Uuid);
  EXPECT_EQ(Spec.GetObjectOffset(), 1024UL);
  EXPECT_EQ(Spec.GetObjectSize(), 3600UL);
  EXPECT_EQ(FileSystem::Instance().GetByteSize(FileSpec(SO)), 4640UL);
}

// Verify an AMDGPU ELF header decodes to its exact GPU model.
// An AMDGPU object with no decodable model resolves to the generic "unknown"
// catch-all.
TEST_F(ObjectFileELFTest, GPUArchitectureUnknownAMDGPUModel) {
  // No model byte is parsed unless the OS ABI is AMDGPU HSA.
  {
    auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  OSABI:           ELFOSABI_NONE
  Type:            ET_DYN
  Machine:         EM_AMDGPU
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000000020
    Size:            0x0000000000000040
...
)");
    ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
    auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
    const ArchSpec &arch = module_sp->GetArchitecture();
    ASSERT_TRUE(arch.IsValid());
    EXPECT_EQ(ArchSpec::eCore_amd_gpu_unknown, arch.GetCore());
  }

  // HSA code-object v2 (ABIVersion 0) carries no model byte, so even a
  // populated mach flag is ignored.
  {
    auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  OSABI:           ELFOSABI_AMDGPU_HSA
  ABIVersion:      0x0
  Type:            ET_DYN
  Machine:         EM_AMDGPU
  Flags:           [ EF_AMDGPU_MACH_AMDGCN_GFX942 ]
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000000020
    Size:            0x0000000000000040
...
)");
    ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
    auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
    const ArchSpec &arch = module_sp->GetArchitecture();
    ASSERT_TRUE(arch.IsValid());
    EXPECT_EQ(ArchSpec::eCore_amd_gpu_unknown, arch.GetCore());
  }
}

namespace {
struct AMDGPUModel {
  uint32_t mach;    // EF_AMDGPU_MACH value.
  const char *flag; // ELF flag name used in the YAML "Flags" field.
  const char *name; // Canonical model name, e.g. "gfx942".
};

// Every AMD GPU model, taken from llvm's AMDGPU_MACH_LIST so the tests track
// the authoritative list instead of duplicating it.
const AMDGPUModel kAMDGPUModels[] = {
#define AMDGPU_MODEL(NUM, ENUM, NAME) {NUM, #ENUM, NAME},
    AMDGPU_MACH_LIST(AMDGPU_MODEL)
#undef AMDGPU_MODEL
};

std::string AMDGPUModelName(const testing::TestParamInfo<AMDGPUModel> &info) {
  // Test names allow only [A-Za-z0-9_]; the generic models contain dashes.
  std::string name = info.param.name;
  for (char &c : name)
    if (c == '-')
      c = '_';
  return name;
}
} // namespace

class ObjectFileELFAMDGPUTest
    : public ObjectFileELFTest,
      public ::testing::WithParamInterface<AMDGPUModel> {};

// Every AMD GPU model must decode from an ELF header (built from YAML) to the
// right arch and core. The model is the EF_AMDGPU_MACH value in e_flags.
TEST_P(ObjectFileELFAMDGPUTest, DecodesAMDGPUModel) {
  const AMDGPUModel &model = GetParam();
  const char *yaml_template = R"(--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  OSABI:           ELFOSABI_AMDGPU_HSA
  ABIVersion:      0x1
  Type:            ET_DYN
  Machine:         EM_AMDGPU
  Flags:           [ {0} ]
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000000020
    Size:            0x0000000000000040
...
)";
  std::string yaml = llvm::formatv(yaml_template, model.flag).str();
  auto ExpectedFile = TestFile::fromYaml(yaml);
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  const ArchSpec &arch = module_sp->GetArchitecture();
  ASSERT_TRUE(arch.IsValid());
  EXPECT_NE(ArchSpec::eCore_amd_gpu_unknown, arch.GetCore());
  EXPECT_EQ(model.name, arch.GetClangTargetCPU());
  bool is_gcn = llvm::StringRef(model.name).starts_with("gfx");
  EXPECT_EQ(is_gcn ? llvm::Triple::amdgpu : llvm::Triple::r600,
            arch.GetTriple().getArch());
  EXPECT_EQ(llvm::Triple::AMD, arch.GetTriple().getVendor());
  EXPECT_EQ(llvm::Triple::AMDHSA, arch.GetTriple().getOS());
}

INSTANTIATE_TEST_SUITE_P(AMDGPUModels, ObjectFileELFAMDGPUTest,
                         ::testing::ValuesIn(kAMDGPUModels), AMDGPUModelName);

TEST_F(ObjectFileELFTest, GetSymtab_SynthesizedEntryPointSymbol) {
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS32
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_SPARC
  Entry:           0x0000000000000042
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000000020
    Size:            0x0000000000000040
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  Symtab *symtab = module_sp->GetSymtab();
  ASSERT_NE(symtab, nullptr);

  // Check that the entry point symbol was synthesized.
  Symbol *entry_sym = symtab->FindSymbolAtFileAddress(0x42);
  ASSERT_TRUE(entry_sym);
  EXPECT_TRUE(entry_sym->IsSyntheticWithAutoGeneratedName());

  // Check that the symbol has the correct section+offset address.
  ASSERT_TRUE(entry_sym->ValueIsAddress());
  Address entry_sym_addr = entry_sym->GetAddress();
  ASSERT_TRUE(entry_sym_addr.IsSectionOffset());
  EXPECT_EQ(entry_sym_addr.GetSection()->GetName(), ".text");
  EXPECT_EQ(entry_sym_addr.GetOffset(), 0x42U - 0x20U);
}

TEST_F(ObjectFileELFTest, GetSymtab_NoSymEntryPointArmThumbAddressClass) {
  /*
  // nosym-entrypoint-arm-thumb.s
  .thumb_func
  _start:
      mov r0, #42
      mov r7, #1
      svc #0
  // arm-linux-androideabi-as nosym-entrypoint-arm-thumb.s
  //   -o nosym-entrypoint-arm-thumb.o
  // arm-linux-androideabi-ld nosym-entrypoint-arm-thumb.o
  //   -o nosym-entrypoint-arm-thumb -e 0x8075 -s
  */
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS32
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_ARM
  Flags:           [ EF_ARM_SOFT_FLOAT, EF_ARM_EABI_VER5 ]
  Entry:           0x0000000000008075
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000008074
    AddressAlign:    0x0000000000000002
    Content:         2A20012700DF
  - Name:            .data
    Type:            SHT_PROGBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x0000000000009000
    AddressAlign:    0x0000000000000001
    Content:         ''
  - Name:            .bss
    Type:            SHT_NOBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x0000000000009000
    AddressAlign:    0x0000000000000001
  - Name:            .note.gnu.gold-version
    Type:            SHT_NOTE
    AddressAlign:    0x0000000000000004
    Content:         040000000900000004000000474E5500676F6C6420312E3131000000
  - Name:            .ARM.attributes
    Type:            SHT_ARM_ATTRIBUTES
    AddressAlign:    0x0000000000000001
    Content:         '4113000000616561626900010900000006020901'
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  auto entry_point_addr = module_sp->GetObjectFile()->GetEntryPointAddress();
  ASSERT_TRUE(entry_point_addr.GetOffset() & 1);
  // Decrease the offsite by 1 to make it into a breakable address since this
  // is Thumb.
  entry_point_addr.Slide(-1);
  ASSERT_EQ(entry_point_addr.GetAddressClass(),
            AddressClass::eCodeAlternateISA);
}

TEST_F(ObjectFileELFTest, GetSymtab_NoSymEntryPointArmAddressClass) {
  /*
  // nosym-entrypoint-arm.s
  _start:
      movs r0, #42
      movs r7, #1
      svc #0
  // arm-linux-androideabi-as nosym-entrypoint-arm.s
  //   -o nosym-entrypoint-arm.o
  // arm-linux-androideabi-ld nosym-entrypoint-arm.o
  //   -o nosym-entrypoint-arm -e 0x8074 -s
  */
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS32
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_ARM
  Flags:           [ EF_ARM_SOFT_FLOAT, EF_ARM_EABI_VER5 ]
  Entry:           0x0000000000008074
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000008074
    AddressAlign:    0x0000000000000004
    Content:         2A00A0E30170A0E3000000EF
  - Name:            .data
    Type:            SHT_PROGBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x0000000000009000
    AddressAlign:    0x0000000000000001
    Content:         ''
  - Name:            .bss
    Type:            SHT_NOBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x0000000000009000
    AddressAlign:    0x0000000000000001
  - Name:            .note.gnu.gold-version
    Type:            SHT_NOTE
    AddressAlign:    0x0000000000000004
    Content:         040000000900000004000000474E5500676F6C6420312E3131000000
  - Name:            .ARM.attributes
    Type:            SHT_ARM_ATTRIBUTES
    AddressAlign:    0x0000000000000001
    Content:         '4113000000616561626900010900000006010801'
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  auto entry_point_addr = module_sp->GetObjectFile()->GetEntryPointAddress();
  ASSERT_EQ(entry_point_addr.GetAddressClass(), AddressClass::eCode);
}

TEST_F(ObjectFileELFTest, SkipsLocalMappingAndDotLSymbols) {
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
  FileHeader:
    Class:           ELFCLASS64
    Data:            ELFDATA2LSB
    Type:            ET_EXEC
    Machine:         EM_RISCV
    Flags:           [ EF_RISCV_RVC, EF_RISCV_FLOAT_ABI_SINGLE ]
    Entry:           0xC0A1B010
  Sections:
    - Name:            .text
      Type:            SHT_PROGBITS
      Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
      Address:         0x0000000000400180
      AddressAlign:    0x0000000000000010
      Content:         554889E5
    - Name:            .data
      Type:            SHT_PROGBITS
      Flags:           [ SHF_WRITE, SHF_ALLOC ]
      Address:         0x0000000000601000
      AddressAlign:    0x0000000000000004
      Content:         2F000000
    - Name:            .riscv.attributes
      Type:            SHT_PROGBITS
      Flags:           [ SHF_ALLOC ]
      Address:         0x0000000000610000
      AddressAlign:    0x0000000000000004
      Content:         "00"
  Symbols:
    - Name:            $d
      Type:            STT_NOTYPE
      Section:         .riscv.attributes
      Value:           0x0000000000400180
      Size:            0x10
      Binding:         STB_LOCAL
    - Name:            $x
      Type:            STT_NOTYPE
      Section:         .text
      Value:           0xC0A1B010
      Size:            0x10
      Binding:         STB_LOCAL
    - Name:            .Lfoo
      Type:            STT_OBJECT
      Section:         .data
      Value:           0x0000000000601000
      Size:            0x4
      Binding:         STB_LOCAL
    - Name:            global_func
      Type:            STT_FUNC
      Section:         .text
      Value:           0x00000000004001A0
      Size:            0x10
      Binding:         STB_GLOBAL
    - Name:            global_obj
      Type:            STT_OBJECT
      Section:         .data
      Value:           0x0000000000601004
      Size:            0x4
      Binding:         STB_GLOBAL
...
  )");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  auto *symtab = module_sp->GetSymtab();
  ASSERT_NE(nullptr, symtab);
  EXPECT_EQ(nullptr, module_sp->FindFirstSymbolWithNameAndType(
                         ConstString("$d"), eSymbolTypeAny));
  EXPECT_EQ(nullptr, module_sp->FindFirstSymbolWithNameAndType(
                         ConstString("$x"), eSymbolTypeAny));
  EXPECT_EQ(nullptr, module_sp->FindFirstSymbolWithNameAndType(
                         ConstString(".Lfoo"), eSymbolTypeAny));
  // assert that other symbols are present
  const Symbol *global_func = module_sp->FindFirstSymbolWithNameAndType(
      ConstString("global_func"), eSymbolTypeAny);
  ASSERT_NE(nullptr, global_func);
  const Symbol *global_obj = module_sp->FindFirstSymbolWithNameAndType(
      ConstString("global_obj"), eSymbolTypeAny);
  ASSERT_NE(nullptr, global_obj);
}

// An ELF filter library (DT_FILTER / DT_AUXILIARY) delegates symbol
// resolution to its filtees.  Its zero-sized exported function definitions
// are placeholders whose addresses are meaningless (e.g. FreeBSD >= 15's
// libc.so.7 exports a zero-sized "mmap" whose address points into an
// unrelated function; the real definition lives in the libsys.so.7 filtee).
// Check that the filtees are reported in dynamic-section order and that the
// symbol table itself is left untouched: marking a symbol as re-exported
// reuses its address range as string-pointer storage (see
// Symbol::SetReExportedSymbolName), which would corrupt the file-address
// indexes for symbols that have a section.  Filtee shadowing happens at
// resolution time instead (see ReExportedSymbolTest.cpp).
TEST_F(ObjectFileELFTest, FilterLibraryPlaceholderSymbols) {
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x1000
    AddressAlign:    0x10
    Content:         C3C3C3C3
# .dynstr content: "\0libaux1.so\0libaux2.so\0"
#   offset 0x1: "libaux1.so"
#   offset 0xC: "libaux2.so"
  - Name:            .dynstr
    Type:            SHT_STRTAB
    Flags:           [ SHF_ALLOC ]
    Address:         0x2000
    AddressAlign:    0x1
    Content:         "006C6962617578312E736F006C6962617578322E736F00"
  - Name:            .dynamic
    Type:            SHT_DYNAMIC
    Flags:           [ SHF_ALLOC ]
    Address:         0x3000
    AddressAlign:    0x8
    Link:            .dynstr
    Entries:
      - Tag:             DT_AUXILIARY
        Value:           0x1
      - Tag:             DT_AUXILIARY
        Value:           0xC
      - Tag:             DT_NULL
        Value:           0x0
Symbols:
  - Name:            local_helper
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1002
    Binding:         STB_LOCAL
  - Name:            mmap
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1000
    Binding:         STB_GLOBAL
  - Name:            "open@@FBSD_1.0"
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1001
    Binding:         STB_WEAK
  - Name:            printf
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1000
    Size:            0x4
    Binding:         STB_GLOBAL
...
  )");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  ObjectFile *object_file = module_sp->GetObjectFile();
  ASSERT_NE(nullptr, object_file);

  // The filtees must come back in dynamic-section order.
  FileSpecList filtees = object_file->GetReExportedLibraries();
  ASSERT_EQ(2u, filtees.GetSize());
  EXPECT_EQ(ConstString("libaux1.so"),
            filtees.GetFileSpecAtIndex(0).GetFilename());
  EXPECT_EQ(ConstString("libaux2.so"),
            filtees.GetFileSpecAtIndex(1).GetFilename());

  // Filtee shadowing is a resolution-time concern: no symbol may be marked
  // eSymbolTypeReExported, which would clobber its address range.
  Symtab *symtab = module_sp->GetSymtab();
  ASSERT_NE(nullptr, symtab);
  for (size_t i = 0, e = symtab->GetNumSymbols(); i != e; ++i) {
    const Symbol *symbol = symtab->SymbolAtIndex(i);
    EXPECT_NE(eSymbolTypeReExported, symbol->GetType())
        << "symbol: " << symbol->GetName().GetStringRef().str();
  }

  // The zero-sized placeholder stays a plain code symbol whose address (and
  // thus the file-address indexes built from it) is intact.
  const Symbol *placeholder = module_sp->FindFirstSymbolWithNameAndType(
      ConstString("mmap"), eSymbolTypeCode);
  ASSERT_NE(nullptr, placeholder);
  EXPECT_TRUE(placeholder->ValueIsAddress());
  EXPECT_EQ(0x1000U, placeholder->GetAddress().GetFileAddress());
  EXPECT_TRUE(placeholder->GetReExportedSymbolName().IsEmpty());

  // A function the filter genuinely defines (nonzero size) keeps its
  // definition, matching the dynamic linker's DT_AUXILIARY fallback.
  const Symbol *real = module_sp->FindFirstSymbolWithNameAndType(
      ConstString("printf"), eSymbolTypeCode);
  ASSERT_NE(nullptr, real);
}

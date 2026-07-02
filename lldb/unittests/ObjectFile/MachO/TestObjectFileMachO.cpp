//===-- ObjectFileMachOTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-defines.h"
#include "gtest/gtest.h"

#ifdef __APPLE__
#include <dlfcn.h>
#endif

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

namespace {
class ObjectFileMachOTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileMachO> subsystems;
};
} // namespace

#if defined(__APPLE__)
TEST_F(ObjectFileMachOTest, ModuleFromSharedCacheInfo) {
  ArchSpec arch("arm64-apple-macosx-");

  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  SharedCacheImageInfo image_info = HostInfo::GetSharedCacheImageInfo(
      ConstString("/usr/lib/libobjc.A.dylib"),
      lldb::eSymbolSharedCacheUseHostSharedCache);
  EXPECT_TRUE(image_info.GetUUID());
  EXPECT_TRUE(image_info.GetExtractor());

  ModuleSpec spec(FileSpec(), UUID(), image_info.GetExtractor());
  lldb::ModuleSP module = std::make_shared<Module>(spec);
  ObjectFile *OF = module->GetObjectFile();
  ASSERT_TRUE(llvm::isa<ObjectFileMachO>(OF));
  EXPECT_TRUE(
      OF->GetArchitecture().IsCompatibleMatch(HostInfo::GetArchitecture()));
  Symtab *symtab = OF->GetSymtab();
  ASSERT_NE(symtab, nullptr);
  void *libobjc = dlopen("/usr/lib/libobjc.A.dylib", RTLD_LAZY);
  ASSERT_NE(libobjc, nullptr);

  // This function checks that if we read something from the
  // ObjectFile we get through the shared cache in-mmeory
  // buffer, it matches what we get by reading directly the
  // memory of the symbol.
  auto check_symbol = [&](const char *sym_name) {
    std::vector<uint32_t> symbol_indices;
    symtab->FindAllSymbolsWithNameAndType(ConstString(sym_name),
                                          lldb::eSymbolTypeAny, symbol_indices);
    EXPECT_EQ(symbol_indices.size(), 1u);

    const Symbol *sym = symtab->SymbolAtIndex(symbol_indices[0]);
    ASSERT_NE(sym, nullptr);
    Address base = sym->GetAddress();
    size_t size = sym->GetByteSize();
    ASSERT_NE(size, 0u);
    uint8_t buffer[size];
    EXPECT_EQ(OF->ReadSectionData(base.GetSection().get(), base.GetOffset(),
                                  buffer, size),
              size);

    void *sym_addr = dlsym(libobjc, sym_name);
    ASSERT_NE(sym_addr, nullptr);
    EXPECT_EQ(memcmp(buffer, sym_addr, size), 0);
  };

  // Read a symbol from the __TEXT segment...
  check_symbol("objc_begin_catch");
  // ... and one from the __DATA segment
  check_symbol("OBJC_IVAR_$_NSObject.isa");
}

TEST_F(ObjectFileMachOTest, IndirectSymbolsInTheSharedCache) {
  SharedCacheImageInfo image_info = HostInfo::GetSharedCacheImageInfo(
      ConstString(
          "/System/Library/Frameworks/AppKit.framework/Versions/C/AppKit"),
      lldb::eSymbolSharedCacheUseHostSharedCache);
  ModuleSpec spec(FileSpec(), UUID(), image_info.GetExtractor());
  lldb::ModuleSP module = std::make_shared<Module>(spec);

  ObjectFile *OF = module->GetObjectFile();
  ASSERT_TRUE(llvm::isa<ObjectFileMachO>(OF));
  EXPECT_TRUE(
      OF->GetArchitecture().IsCompatibleMatch(HostInfo::GetArchitecture()));

  // Check that we can parse the symbol table several times over without
  // crashing.
  Symtab symtab(OF);
  for (size_t i = 0; i < 10; i++)
    OF->ParseSymtab(symtab);
}
#endif

// Regression fixture: a Mach-O whose load commands all have cmdsize = 0.
// With ncmds set near INT_MAX the functions hang.  The fix breaks out of
// the loop as soon as cmdsize < sizeof(load_command).  Found by
// lldb-target-fuzzer.
TEST_F(ObjectFileMachOTest, ZeroCmdSize) {
  // Minimal little-endian x86_64 Mach-O: mach_header_64 (32 bytes) followed
  // by a single load_command with cmdsize = 0.  ncmds is set to 0x7FFFFFFF so
  // that without the fix the loops in GetAllArchSpecs/GetUUID never return in
  // practice; with the fix they break on the very first iteration.
  // Reaching the assertion below is the regression check.
  const uint8_t kData[] = {
      // mach_header_64 (little-endian)
      0xCF,
      0xFA,
      0xED,
      0xFE, // magic:      MH_MAGIC_64
      0x07,
      0x00,
      0x00,
      0x01, // cputype:    CPU_TYPE_X86_64
      0x03,
      0x00,
      0x00,
      0x80, // cpusubtype: CPU_SUBTYPE_X86_64_ALL
      0x02,
      0x00,
      0x00,
      0x00, // filetype:   MH_EXECUTE
      0xFF,
      0xFF,
      0xFF,
      0x7F, // ncmds:      0x7FFFFFFF
      0x08,
      0x00,
      0x00,
      0x00, // sizeofcmds: 8
      0x00,
      0x00,
      0x00,
      0x00, // flags:      0
      0x00,
      0x00,
      0x00,
      0x00, // reserved:   0
      // load_command
      0x19,
      0x00,
      0x00,
      0x00, // cmd:     LC_SEGMENT_64 (arbitrary)
      0x00,
      0x00,
      0x00,
      0x00, // cmdsize: 0  ← causes the spin
  };
  auto Buf = std::make_shared<DataBufferHeap>(kData, sizeof(kData));
  lldb::DataExtractorSP DataSP = std::make_shared<lldb_private::DataExtractor>(
      Buf, lldb::eByteOrderLittle, /*addr_size=*/8);
  // Before the fix GetAllArchSpecs loops ~0x7FFFFFFF times and never returns.
  (void)ObjectFile::GetModuleSpecifications(FileSpec(), DataSP, 0,
                                            sizeof(kData));
}

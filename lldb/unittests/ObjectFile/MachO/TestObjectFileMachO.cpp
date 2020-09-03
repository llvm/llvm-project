//===-- ObjectFileMachOTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/HostInfo.h"
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/lldb-defines.h"
#include "gtest/gtest.h"

#ifdef __APPLE__
#include <dlfcn.h>
#endif

using namespace lldb_private;
using namespace llvm;

namespace {
class ObjectFileMachOTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileMachO> subsystems;
};
} // namespace

#if defined(__APPLE__)
TEST_F(ObjectFileMachOTest, ModuleFromSharedCacheInfo) {
  SharedCacheImageInfo image_info =
      HostInfo::GetSharedCacheImageInfo("/usr/lib/libobjc.A.dylib");
  EXPECT_TRUE(image_info.uuid);
  EXPECT_TRUE(image_info.data_sp);

  ModuleSpec spec(FileSpec(), UUID(), image_info.data_sp);
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

    Symbol *sym = symtab->SymbolAtIndex(symbol_indices[0]);
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
  check_symbol("objc_msgSend");
  // ... and one from the __DATA segment
  check_symbol("OBJC_CLASS_$_NSObject");
}
#endif

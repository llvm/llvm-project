//===- llvm/unittest/BinaryFormat/TestFileMagic.cpp - File magic tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "gtest/gtest.h"

using namespace llvm;
namespace fs = llvm::sys::fs;

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    SmallString<128> MessageStorage;                                           \
    raw_svector_ostream Message(MessageStorage);                               \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }

class MagicTest : public testing::Test {
protected:
  /// Unique temporary directory in which all created filesystem entities must
  /// be placed. It is removed at the end of each test (must be empty).
  SmallString<128> TestDirectory;

  void SetUp() override {
    ASSERT_NO_ERROR(
        fs::createUniqueDirectory("file-system-test", TestDirectory));
    // We don't care about this specific file.
    errs() << "Test Directory: " << TestDirectory << '\n';
    errs().flush();
  }

  void TearDown() override { ASSERT_NO_ERROR(fs::remove(TestDirectory.str())); }
};

const char archive[] = "!<arch>\x0A";
const char bitcode[] = "\xde\xc0\x17\x0b";
const char coff_object[] = "\x00\x00......";
const char coff_bigobj[] =
    "\x00\x00\xff\xff\x00\x02......"
    "\xc7\xa1\xba\xd1\xee\xba\xa9\x4b\xaf\x20\xfa\xf6\x6a\xa4\xdc\xb8";
const char coff_import_library[] = "\x00\x00\xff\xff....";
const char elf_relocatable[] = {0x7f, 'E', 'L', 'F', 1, 2, 1, 0, 0,
                                0,    0,   0,   0,   0, 0, 0, 0, 1};

const char goff_object[] = "\x03\xF0\x00";
const char macho_universal_binary[] = "\xca\xfe\xba\xbe...\x00";
const char macho_object[] =
    "\xfe\xed\xfa\xce........\x00\x00\x00\x01............";
const char macho_executable[] =
    "\xfe\xed\xfa\xce........\x00\x00\x00\x02............";
const char macho_fixed_virtual_memory_shared_lib[] =
    "\xfe\xed\xfa\xce........\x00\x00\x00\x03............";
const char macho_core[] =
    "\xfe\xed\xfa\xce........\x00\x00\x00\x04............";
const char macho_preload_executable[] =
    "\xfe\xed\xfa\xce........\x00\x00\x00\x05............";
const char macho_dynamically_linked_shared_lib[] =
    "\xfe\xed\xfa\xce........\x00\x00\x00\x06............";
const char macho_dynamic_linker[] =
    "\xfe\xed\xfa\xce........\x00\x00\x00\x07............";
const char macho_bundle[] =
    "\xfe\xed\xfa\xce........\x00\x00\x00\x08............";
const char macho_dsym_companion[] =
    "\xfe\xed\xfa\xce........\x00\x00\x00\x0a............";
const char macho_kext_bundle[] =
    "\xfe\xed\xfa\xce........\x00\x00\x00\x0b............";
const char windows_resource[] =
    "\x00\x00\x00\x00\x020\x00\x00\x00\xff\xff\x00\x00\xff\xff\x00\x00";
const char macho_dynamically_linked_shared_lib_stub[] =
    "\xfe\xed\xfa\xce........\x00\x00\x00\x09............";
const char ms_dos_stub_broken[] = "\x4d\x5a\x20\x20";
const char pdb[] = "Microsoft C/C++ MSF 7.00\r\n\x1a"
                   "DS\x00\x00\x00";
const char tapi_file[] = "--- !tapi-tbd-v1\n";
const char tapi_file_tbd_v1[] = "---\narchs: [";

TEST_F(MagicTest, Magic) {
  struct type {
    const char *filename;
    const char *magic_str;
    size_t magic_str_len;
    file_magic magic;
  } types[] = {
#define DEFINE(magic) {#magic, magic, sizeof(magic), file_magic::magic}
      DEFINE(archive),
      DEFINE(bitcode),
      DEFINE(coff_object),
      {"coff_bigobj", coff_bigobj, sizeof(coff_bigobj),
       file_magic::coff_object},
      DEFINE(coff_import_library),
      DEFINE(elf_relocatable),
      DEFINE(goff_object),
      DEFINE(macho_universal_binary),
      DEFINE(macho_object),
      DEFINE(macho_executable),
      DEFINE(macho_fixed_virtual_memory_shared_lib),
      DEFINE(macho_core),
      DEFINE(macho_preload_executable),
      DEFINE(macho_dynamically_linked_shared_lib),
      DEFINE(macho_dynamic_linker),
      DEFINE(macho_bundle),
      DEFINE(macho_dynamically_linked_shared_lib_stub),
      DEFINE(macho_dsym_companion),
      DEFINE(macho_kext_bundle),
      DEFINE(windows_resource),
      DEFINE(pdb),
      {"ms_dos_stub_broken", ms_dos_stub_broken, sizeof(ms_dos_stub_broken),
       file_magic::unknown},
      DEFINE(tapi_file),
      {"tapi_file_tbd_v1", tapi_file_tbd_v1, sizeof(tapi_file_tbd_v1),
       file_magic::tapi_file},
#undef DEFINE
  };

  // Create some files filled with magic.
  for (type *i = types, *e = types + (sizeof(types) / sizeof(type)); i != e;
       ++i) {
    SmallString<128> file_pathname(TestDirectory);
    llvm::sys::path::append(file_pathname, i->filename);
    std::error_code EC;
    raw_fd_ostream file(file_pathname, EC, sys::fs::OF_None);
    ASSERT_FALSE(file.has_error());
    StringRef magic(i->magic_str, i->magic_str_len);
    file << magic;
    file.close();
    EXPECT_EQ(i->magic, identify_magic(magic));
    ASSERT_NO_ERROR(fs::remove(Twine(file_pathname)));
  }
}

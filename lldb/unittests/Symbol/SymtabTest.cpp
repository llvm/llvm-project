//===-- SymtabTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/DataFileCache.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/DataExtractor.h"

#include <memory>

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;

class SymtabTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileMachO, SymbolFileDWARF,
                ObjectFileELF, SymbolFileSymtab, TypeSystemClang>
      subsystem;
};

static void EncodeDecode(const Symtab &object, ByteOrder byte_order) {
  const uint8_t addr_size = 8;
  DataEncoder file(byte_order, addr_size);

  object.Encode(file);
  llvm::ArrayRef<uint8_t> bytes = file.GetData();
  DataExtractor data(bytes.data(), bytes.size(), byte_order, addr_size);
  Symtab decoded_object(object.GetObjectFile());
  offset_t data_offset = 0;
  bool uuid_mismatch = false;
  decoded_object.Decode(data, &data_offset, uuid_mismatch);
  ASSERT_EQ(object.GetNumSymbols(), decoded_object.GetNumSymbols());
  for (size_t i = 0; i < object.GetNumSymbols(); ++i)
    EXPECT_EQ(*object.SymbolAtIndex(i), *decoded_object.SymbolAtIndex(i));
}

static void EncodeDecode(const Symtab &object) {
  EncodeDecode(object, eByteOrderLittle);
  EncodeDecode(object, eByteOrderBig);
}

TEST_F(SymtabTest, EncodeDecodeSymtab) {

  auto ExpectedFile = TestFile::fromYaml(R"(
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x100000C
  cpusubtype:      0x0
  filetype:        0x2
  ncmds:           17
  sizeofcmds:      792
  flags:           0x200085
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         72
    segname:         __PAGEZERO
    vmaddr:          0
    vmsize:          4294967296
    fileoff:         0
    filesize:        0
    maxprot:         0
    initprot:        0
    nsects:          0
    flags:           0
  - cmd:             LC_SEGMENT_64
    cmdsize:         232
    segname:         __TEXT
    vmaddr:          4294967296
    vmsize:          16384
    fileoff:         0
    filesize:        16384
    maxprot:         5
    initprot:        5
    nsects:          2
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x100003F94
        size:            36
        offset:          0x3F94
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         FF8300D1E80300AA00008052FF1F00B9E81B00B9E10B00F9E20700F9FF830091C0035FD6
      - sectname:        __unwind_info
        segname:         __TEXT
        addr:            0x100003FB8
        size:            72
        offset:          0x3FB8
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         010000001C000000000000001C000000000000001C00000002000000943F00003400000034000000B93F00000000000034000000030000000C000100100001000000000000200002
  - cmd:             LC_SEGMENT_64
    cmdsize:         72
    segname:         __LINKEDIT
    vmaddr:          4294983680
    vmsize:          16384
    fileoff:         16384
    filesize:        674
    maxprot:         1
    initprot:        1
    nsects:          0
    flags:           0
  - cmd:             LC_DYLD_CHAINED_FIXUPS
    cmdsize:         16
    dataoff:         16384
    datasize:        56
  - cmd:             LC_DYLD_EXPORTS_TRIE
    cmdsize:         16
    dataoff:         16440
    datasize:        48
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          16496
    nsyms:           10
    stroff:          16656
    strsize:         128
  - cmd:             LC_DYSYMTAB
    cmdsize:         80
    ilocalsym:       0
    nlocalsym:       8
    iextdefsym:      8
    nextdefsym:      2
    iundefsym:       10
    nundefsym:       0
    tocoff:          0
    ntoc:            0
    modtaboff:       0
    nmodtab:         0
    extrefsymoff:    0
    nextrefsyms:     0
    indirectsymoff:  0
    nindirectsyms:   0
    extreloff:       0
    nextrel:         0
    locreloff:       0
    nlocrel:         0
  - cmd:             LC_LOAD_DYLINKER
    cmdsize:         32
    name:            12
    Content:         '/usr/lib/dyld'
    ZeroPadBytes:    7
  - cmd:             LC_UUID
    cmdsize:         24
    uuid:            1EECD2B8-16EA-3FEC-AB3C-F46139DBD0E2
  - cmd:             LC_BUILD_VERSION
    cmdsize:         32
    platform:        1
    minos:           786432
    sdk:             786432
    ntools:          1
    Tools:
      - tool:            3
        version:         46596096
  - cmd:             LC_SOURCE_VERSION
    cmdsize:         16
    version:         0
  - cmd:             LC_MAIN
    cmdsize:         24
    entryoff:        16276
    stacksize:       0
  - cmd:             LC_LOAD_DYLIB
    cmdsize:         48
    dylib:
      name:            24
      timestamp:       2
      current_version: 78643968
      compatibility_version: 65536
    Content:         '/usr/lib/libc++.1.dylib'
    ZeroPadBytes:    1
  - cmd:             LC_LOAD_DYLIB
    cmdsize:         56
    dylib:
      name:            24
      timestamp:       2
      current_version: 85917696
      compatibility_version: 65536
    Content:         '/usr/lib/libSystem.B.dylib'
    ZeroPadBytes:    6
  - cmd:             LC_FUNCTION_STARTS
    cmdsize:         16
    dataoff:         16488
    datasize:        8
  - cmd:             LC_DATA_IN_CODE
    cmdsize:         16
    dataoff:         16496
    datasize:        0
  - cmd:             LC_CODE_SIGNATURE
    cmdsize:         16
    dataoff:         16784
    datasize:        274
LinkEditData:
  NameList:
    - n_strx:          28
      n_type:          0x64
      n_sect:          0
      n_desc:          0
      n_value:         0
    - n_strx:          64
      n_type:          0x64
      n_sect:          0
      n_desc:          0
      n_value:         0
    - n_strx:          73
      n_type:          0x66
      n_sect:          0
      n_desc:          1
      n_value:         1639532873
    - n_strx:          1
      n_type:          0x2E
      n_sect:          1
      n_desc:          0
      n_value:         4294983572
    - n_strx:          115
      n_type:          0x24
      n_sect:          1
      n_desc:          0
      n_value:         4294983572
    - n_strx:          1
      n_type:          0x24
      n_sect:          0
      n_desc:          0
      n_value:         36
    - n_strx:          1
      n_type:          0x4E
      n_sect:          1
      n_desc:          0
      n_value:         36
    - n_strx:          1
      n_type:          0x64
      n_sect:          1
      n_desc:          0
      n_value:         0
    - n_strx:          2
      n_type:          0xF
      n_sect:          1
      n_desc:          16
      n_value:         4294967296
    - n_strx:          22
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         4294983572
  StringTable:
    - ' '
    - __mh_execute_header
    - _main
    - '/Users/gclayton/Documents/src/args/'
    - main.cpp
    - '/Users/gclayton/Documents/src/args/main.o'
    - _main
    - ''
    - ''
    - ''
    - ''
    - ''
    - ''
    - ''
...
)");

  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  ObjectFile *objfile = module_sp->GetObjectFile();
  ASSERT_NE(objfile, nullptr);

  // Test encoding and decoding an empty symbol table.
  Symtab symtab(objfile);
  symtab.PreloadSymbols();
  EncodeDecode(symtab);

  // Now encode and decode an actual symbol table from our yaml.
  Symtab *module_symtab = module_sp->GetSymtab();
  ASSERT_NE(module_symtab, nullptr);
  module_symtab->PreloadSymbols();
  EncodeDecode(*module_symtab);
}

TEST_F(SymtabTest, TestDecodeCStringMaps) {
  // Symbol tables save out the symbols, but they also save out the symbol table
  // name indexes. These name indexes are a map of sorted ConstString + T pairs
  // and when they are decoded from a file, they are no longer sorted since
  // ConstString objects can be sorted by "const char *" and the order in which
  // these strings are created won't be the same in a new process. We need to
  // ensure these name lookups happen correctly when we load the name indexes,
  // so this test loads a symbol table from a cache file from
  // "lldb/unittests/Symbol/Inputs/indexnames-symtab-cache" and make sure we
  // can correctly lookup each of the names in the symbol table.
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x100000C
  cpusubtype:      0x0
  filetype:        0x2
  ncmds:           16
  sizeofcmds:      744
  flags:           0x200085
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         72
    segname:         __PAGEZERO
    vmaddr:          0
    vmsize:          4294967296
    fileoff:         0
    filesize:        0
    maxprot:         0
    initprot:        0
    nsects:          0
    flags:           0
  - cmd:             LC_SEGMENT_64
    cmdsize:         232
    segname:         __TEXT
    vmaddr:          4294967296
    vmsize:          16384
    fileoff:         0
    filesize:        16384
    maxprot:         5
    initprot:        5
    nsects:          2
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x100003F64
        size:            76
        offset:          0x3F64
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         80018052C0035FD6E0028052C0035FD640048052C0035FD6FF8300D1FD7B01A9FD43009108008052E80B00B9BFC31FB8F4FFFF97F5FFFF97F6FFFF97E00B40B9FD7B41A9FF830091C0035FD6
      - sectname:        __unwind_info
        segname:         __TEXT
        addr:            0x100003FB0
        size:            80
        offset:          0x3FB0
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         010000001C000000000000001C000000000000001C00000002000000643F00003400000034000000B13F00000000000034000000030000000C0002001400020000000001180000000000000400000002
  - cmd:             LC_SEGMENT_64
    cmdsize:         72
    segname:         __LINKEDIT
    vmaddr:          4294983680
    vmsize:          16384
    fileoff:         16384
    filesize:        994
    maxprot:         1
    initprot:        1
    nsects:          0
    flags:           0
  - cmd:             LC_DYLD_CHAINED_FIXUPS
    cmdsize:         16
    dataoff:         16384
    datasize:        56
  - cmd:             LC_DYLD_EXPORTS_TRIE
    cmdsize:         16
    dataoff:         16440
    datasize:        80
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          16528
    nsyms:           25
    stroff:          16928
    strsize:         176
  - cmd:             LC_DYSYMTAB
    cmdsize:         80
    ilocalsym:       0
    nlocalsym:       20
    iextdefsym:      20
    nextdefsym:      5
    iundefsym:       25
    nundefsym:       0
    tocoff:          0
    ntoc:            0
    modtaboff:       0
    nmodtab:         0
    extrefsymoff:    0
    nextrefsyms:     0
    indirectsymoff:  0
    nindirectsyms:   0
    extreloff:       0
    nextrel:         0
    locreloff:       0
    nlocrel:         0
  - cmd:             LC_LOAD_DYLINKER
    cmdsize:         32
    name:            12
    Content:         '/usr/lib/dyld'
    ZeroPadBytes:    7
  - cmd:             LC_UUID
    cmdsize:         24
    uuid:            3E94866E-0D1A-39BD-975B-64E8F1FDBAAE
  - cmd:             LC_BUILD_VERSION
    cmdsize:         32
    platform:        1
    minos:           786432
    sdk:             787200
    ntools:          1
    Tools:
      - tool:            3
        version:         49938432
  - cmd:             LC_SOURCE_VERSION
    cmdsize:         16
    version:         0
  - cmd:             LC_MAIN
    cmdsize:         24
    entryoff:        16252
    stacksize:       0
  - cmd:             LC_LOAD_DYLIB
    cmdsize:         56
    dylib:
      name:            24
      timestamp:       2
      current_version: 85943299
      compatibility_version: 65536
    Content:         '/usr/lib/libSystem.B.dylib'
    ZeroPadBytes:    6
  - cmd:             LC_FUNCTION_STARTS
    cmdsize:         16
    dataoff:         16520
    datasize:        8
  - cmd:             LC_DATA_IN_CODE
    cmdsize:         16
    dataoff:         16528
    datasize:        0
  - cmd:             LC_CODE_SIGNATURE
    cmdsize:         16
    dataoff:         17104
    datasize:        274
LinkEditData:
  NameList:
    - n_strx:          43
      n_type:          0x64
      n_sect:          0
      n_desc:          0
      n_value:         0
    - n_strx:          91
      n_type:          0x64
      n_sect:          0
      n_desc:          0
      n_value:         0
    - n_strx:          98
      n_type:          0x66
      n_sect:          0
      n_desc:          1
      n_value:         1651098491
    - n_strx:          1
      n_type:          0x2E
      n_sect:          1
      n_desc:          0
      n_value:         4294983524
    - n_strx:          152
      n_type:          0x24
      n_sect:          1
      n_desc:          0
      n_value:         4294983524
    - n_strx:          1
      n_type:          0x24
      n_sect:          0
      n_desc:          0
      n_value:         8
    - n_strx:          1
      n_type:          0x4E
      n_sect:          1
      n_desc:          0
      n_value:         8
    - n_strx:          1
      n_type:          0x2E
      n_sect:          1
      n_desc:          0
      n_value:         4294983532
    - n_strx:          157
      n_type:          0x24
      n_sect:          1
      n_desc:          0
      n_value:         4294983532
    - n_strx:          1
      n_type:          0x24
      n_sect:          0
      n_desc:          0
      n_value:         8
    - n_strx:          1
      n_type:          0x4E
      n_sect:          1
      n_desc:          0
      n_value:         8
    - n_strx:          1
      n_type:          0x2E
      n_sect:          1
      n_desc:          0
      n_value:         4294983540
    - n_strx:          162
      n_type:          0x24
      n_sect:          1
      n_desc:          0
      n_value:         4294983540
    - n_strx:          1
      n_type:          0x24
      n_sect:          0
      n_desc:          0
      n_value:         8
    - n_strx:          1
      n_type:          0x4E
      n_sect:          1
      n_desc:          0
      n_value:         8
    - n_strx:          1
      n_type:          0x2E
      n_sect:          1
      n_desc:          0
      n_value:         4294983548
    - n_strx:          167
      n_type:          0x24
      n_sect:          1
      n_desc:          0
      n_value:         4294983548
    - n_strx:          1
      n_type:          0x24
      n_sect:          0
      n_desc:          0
      n_value:         52
    - n_strx:          1
      n_type:          0x4E
      n_sect:          1
      n_desc:          0
      n_value:         52
    - n_strx:          1
      n_type:          0x64
      n_sect:          1
      n_desc:          0
      n_value:         0
    - n_strx:          2
      n_type:          0xF
      n_sect:          1
      n_desc:          16
      n_value:         4294967296
    - n_strx:          22
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         4294983532
    - n_strx:          27
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         4294983540
    - n_strx:          32
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         4294983524
    - n_strx:          37
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         4294983548
  StringTable:
    - ' '
    - __mh_execute_header
    - _bar
    - _baz
    - _foo
    - _main
    - '/Users/gclayton/Documents/objfiles/index-names/'
    - main.c
    - '/Users/gclayton/Documents/objfiles/index-names/main.o'
    - _foo
    - _bar
    - _baz
    - _main
    - ''
    - ''
    - ''
  FunctionStarts:  [ 0x3F64, 0x3F6C, 0x3F74, 0x3F7C ]
...
)");
  // This data was taken from a hex dump of the object file from the above yaml
  // and hexdumped so we can load the cache data in this test.
  const uint8_t symtab_cache_bytes[] = {
      0x01, 0x10, 0x3e, 0x94, 0x86, 0x6e, 0x0d, 0x1a, 0x39, 0xbd, 0x97, 0x5b,
      0x64, 0xe8, 0xf1, 0xfd, 0xba, 0xae, 0xff, 0x53, 0x54, 0x41, 0x42, 0x91,
      0x00, 0x00, 0x00, 0x00, 0x2f, 0x55, 0x73, 0x65, 0x72, 0x73, 0x2f, 0x67,
      0x63, 0x6c, 0x61, 0x79, 0x74, 0x6f, 0x6e, 0x2f, 0x44, 0x6f, 0x63, 0x75,
      0x6d, 0x65, 0x6e, 0x74, 0x73, 0x2f, 0x6f, 0x62, 0x6a, 0x66, 0x69, 0x6c,
      0x65, 0x73, 0x2f, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x2d, 0x6e, 0x61, 0x6d,
      0x65, 0x73, 0x2f, 0x6d, 0x61, 0x69, 0x6e, 0x2e, 0x63, 0x00, 0x2f, 0x55,
      0x73, 0x65, 0x72, 0x73, 0x2f, 0x67, 0x63, 0x6c, 0x61, 0x79, 0x74, 0x6f,
      0x6e, 0x2f, 0x44, 0x6f, 0x63, 0x75, 0x6d, 0x65, 0x6e, 0x74, 0x73, 0x2f,
      0x6f, 0x62, 0x6a, 0x66, 0x69, 0x6c, 0x65, 0x73, 0x2f, 0x69, 0x6e, 0x64,
      0x65, 0x78, 0x2d, 0x6e, 0x61, 0x6d, 0x65, 0x73, 0x2f, 0x6d, 0x61, 0x69,
      0x6e, 0x2e, 0x6f, 0x00, 0x66, 0x6f, 0x6f, 0x00, 0x62, 0x61, 0x72, 0x00,
      0x62, 0x61, 0x7a, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x5f, 0x6d, 0x68,
      0x5f, 0x65, 0x78, 0x65, 0x63, 0x75, 0x74, 0x65, 0x5f, 0x68, 0x65, 0x61,
      0x64, 0x65, 0x72, 0x00, 0x53, 0x59, 0x4d, 0x42,
      // clang-format off
      0x04, 0x00, 0x00, 0x00, // DataFileCache's CURRENT_CACHE_VERSION
      // clang-format on
      0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x2a,
      0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x64, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x20, 0x01, 0x37,
      0x00, 0x00, 0x00, 0x00, 0x7b, 0xc3, 0x69, 0x62, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x66, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x32, 0x01, 0x6d, 0x00, 0x00,
      0x00, 0x01, 0x64, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x08, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x32, 0x01, 0x71, 0x00, 0x00, 0x00, 0x01,
      0x6c, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x0c, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x02, 0x32, 0x01, 0x75, 0x00, 0x00, 0x00, 0x01, 0x74, 0x3f,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x02, 0x32, 0x01, 0x79, 0x00, 0x00, 0x00, 0x01, 0x7c, 0x3f, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x0f, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x12,
      0x01, 0x7e, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x64, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
      0x0f, 0x00, 0x01, 0x00, 0x43, 0x4d, 0x41, 0x50, 0x07, 0x00, 0x00, 0x00,
      0x71, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x79, 0x00, 0x00, 0x00,
      0x05, 0x00, 0x00, 0x00, 0x6d, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x7e, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x37, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x75, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00};

  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  ObjectFile *objfile = module_sp->GetObjectFile();
  ASSERT_NE(objfile, nullptr);

  // Test encoding and decoding an empty symbol table.
  DataExtractor data(symtab_cache_bytes, sizeof(symtab_cache_bytes),
                     eByteOrderLittle, 8);
  Symtab symtab(objfile);
  offset_t data_offset = 0;
  bool uuid_mismatch = false; // Gets set to true if signature doesn't match.
  const bool success = symtab.Decode(data, &data_offset, uuid_mismatch);
  ASSERT_EQ(success, true);
  ASSERT_EQ(uuid_mismatch, false);

  // Now make sure that name lookup works for all symbols. This indicates that
  // the Symtab::NameToIndexMap was decoded correctly and works as expected.
  const Symbol *symbol = nullptr;
  symbol = symtab.FindFirstSymbolWithNameAndType(ConstString("main"),
                                                 eSymbolTypeCode,
                                                 Symtab::eDebugAny,
                                                 Symtab::eVisibilityAny);
  ASSERT_NE(symbol, nullptr);
  symbol = symtab.FindFirstSymbolWithNameAndType(ConstString("foo"),
                                                 eSymbolTypeCode,
                                                 Symtab::eDebugAny,
                                                 Symtab::eVisibilityAny);
  ASSERT_NE(symbol, nullptr);
  symbol = symtab.FindFirstSymbolWithNameAndType(ConstString("bar"),
                                                 eSymbolTypeCode,
                                                 Symtab::eDebugAny,
                                                 Symtab::eVisibilityAny);
  ASSERT_NE(symbol, nullptr);
  symbol = symtab.FindFirstSymbolWithNameAndType(ConstString("baz"),
                                                 eSymbolTypeCode,
                                                 Symtab::eDebugAny,
                                                 Symtab::eVisibilityAny);
  ASSERT_NE(symbol, nullptr);
}

TEST_F(SymtabTest, TestSymbolFileCreatedOnDemand) {
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
  Entry:           0x0000000000400180
Sections:
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
Symbols:
  - Name:            _start
    Type:            STT_FUNC
    Section:         .text
    Value:           0x0000000000400180
    Size:            0x0000000000000014
    Binding:         STB_GLOBAL
  - Name:            global_data
    Type:            STT_OBJECT
    Section:         .data
    Value:           0x0000000000601000
    Size:            0x0000000000000004
    Binding:         STB_GLOBAL
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  // The symbol file should not be created by default.
  Symtab *module_symtab = module_sp->GetSymtab(/*can_create=*/false);
  ASSERT_EQ(module_symtab, nullptr);

  // But it should be created on demand.
  module_symtab = module_sp->GetSymtab(/*can_create=*/true);
  ASSERT_NE(module_symtab, nullptr);
  ASSERT_NE(module_symtab->FindFirstSymbolWithNameAndType(
                ConstString("_start"), eSymbolTypeCode, Symtab::eDebugAny,
                Symtab::eVisibilityAny),
            nullptr);
  ASSERT_NE(module_symtab->FindFirstSymbolWithNameAndType(
                ConstString("global_data"), eSymbolTypeData, Symtab::eDebugAny,
                Symtab::eVisibilityAny),
            nullptr);

  // And we should be able to get it again once it has been created.
  Symtab *cached_module_symtab = module_sp->GetSymtab(/*can_create=*/false);
  ASSERT_EQ(module_symtab, cached_module_symtab);

  SymbolFile *symbol_file = module_sp->GetSymbolFile();
  ASSERT_NE(symbol_file, nullptr);
  EXPECT_EQ(symbol_file->GetPluginName(),
            SymbolFileSymtab::GetPluginNameStatic());
  EXPECT_EQ(symbol_file->GetAbilities(),
            static_cast<uint32_t>(SymbolFile::Symbols));
}

TEST_F(SymtabTest, TestSymbolTableCreatedOnDemand) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_addr_base
              Form:            DW_FORM_sec_offset
  debug_info:
    - Version:         5
      AddrSize:        4
      UnitType:        DW_UT_compile
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x8 # Offset of the first Address past the header
        - AbbrCode:        0x0
  debug_addr:
    - Version: 5
      AddressSize: 4
      Entries:
        - Address: 0x1234
        - Address: 0x5678
  debug_line:
    - Length:          42
      Version:         2
      PrologueLength:  36
      MinInstLength:   1
      DefaultIsStmt:   1
      LineBase:        251
      LineRange:       14
      OpcodeBase:      13
      StandardOpcodeLengths: [ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ]
      IncludeDirs:
        - '/tmp'
      Files:
        - Name:            main.cpp
          DirIdx:          1
          ModTime:         0
          Length:          0
)";
  llvm::Expected<TestFile> file = TestFile::fromYaml(yamldata);
  EXPECT_THAT_EXPECTED(file, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(file->moduleSpec());

  SymbolFile *symbol_file = module_sp->GetSymbolFile();
  // At this point, the symbol table is not created. This is because the above
  // yaml data contains the necessary sections in order for
  // SymbolFileDWARF::CalculateAbilities() to identify all abilities, saving it
  // from calling SymbolFileDWARFDebugMap::CalculateAbilities(), which
  // eventually loads the symbol table, which we don't want.

  // The symbol table should not be created if asked not to.
  Symtab *symtab = symbol_file->GetSymtab(/*can_create=*/false);
  ASSERT_EQ(symtab, nullptr);

  // But it should be created on demand.
  symtab = symbol_file->GetSymtab(/*can_create=*/true);
  ASSERT_NE(symtab, nullptr);

  // And we should be able to get it again once it has been created.
  Symtab *cached_symtab = symbol_file->GetSymtab(/*can_create=*/false);
  ASSERT_EQ(symtab, cached_symtab);
}

TEST_F(SymtabTest, TestSymbolTableRelativeFileAddresses) {
  // Section-relative in a serialized symbol table are expressed as
  // offsets from the base file address of the binary.  It is possible
  // for a binary to be zero-address offset initially, but then included
  // in a Darwin shared cache with all of the segment and file addresses
  // slid to a non-zero value -- with the same filename and UUID.  To make
  // the serialized symtab work for both, we must make the file addresses
  // relative ot the file base.
  //
  // static int c = 10;
  // int d = 20;
  // int foo() { return c; }
  // int bar() { return foo() + d; }
  // int main() { int var = bar() + c; return var; }
  //
  // clang -dynamiclib -mmacosx-version-min=11 -o liba.dylib a.c
  // obj2yaml liba.dylib > zero.yaml
  // clang -image_base 0x3c000 -dynamiclib -mmacosx-version-min=11 -o liba.dylib
  // a.c obj2yaml liba.dylib > nonzero.yaml
  //
  // hand-edit the UUID from one of them to match the other.
  // Any macOS version higher than 11 results in chained fixups in the binary,
  // and you can't compile the dylib to a non-zero offset with chained fixups.
  //
  // These two YAMLs are below are then serialized into a buffer, and we
  // compare the buffers -- they must be byte-by-byte identical.

  auto ExpectedFileZeroOffset = TestFile::fromYaml(R"(
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x100000C
  cpusubtype:      0x0
  filetype:        0x6
  ncmds:           14
  sizeofcmds:      824
  flags:           0x100085
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         232
    segname:         __TEXT
    vmaddr:          0
    vmsize:          16384
    fileoff:         0
    filesize:        16384
    maxprot:         5
    initprot:        5
    nsects:          2
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x388
        size:            96
        offset:          0x388
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         28000090000540B9C0035FD6FD7BBFA9FD030091FBFFFF9728000090080140B90000080BFD7BC1A8C0035FD6FF8300D1FD7B01A9FD430091BFC31FB8F4FFFF9728000090080540B90800080BE80B00B9E00B40B9FD7B41A9FF830091C0035FD6
      - sectname:        __unwind_info
        segname:         __TEXT
        addr:            0x3E8
        size:            96
        offset:          0x3E8
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         010000001C000000000000001C000000000000001C00000002000000880300004000000040000000E90300000000000040000000000000000000000000000000030000000C00020014000200000000000C000001000000020000000400000000
  - cmd:             LC_SEGMENT_64
    cmdsize:         152
    segname:         __DATA
    vmaddr:          16384
    vmsize:          16384
    fileoff:         16384
    filesize:        16384
    maxprot:         3
    initprot:        3
    nsects:          1
    flags:           0
    Sections:
      - sectname:        __data
        segname:         __DATA
        addr:            0x4000
        size:            8
        offset:          0x4000
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         140000000A000000
  - cmd:             LC_SEGMENT_64
    cmdsize:         72
    segname:         __LINKEDIT
    vmaddr:          32768
    vmsize:          16384
    fileoff:         32768
    filesize:        584
    maxprot:         1
    initprot:        1
    nsects:          0
    flags:           0
  - cmd:             LC_ID_DYLIB
    cmdsize:         40
    dylib:
      name:            24
      timestamp:       1
      current_version: 0
      compatibility_version: 0
    Content:         liba.dylib
    ZeroPadBytes:    6
  - cmd:             LC_DYLD_INFO_ONLY
    cmdsize:         48
    rebase_off:      0
    rebase_size:     0
    bind_off:        0
    bind_size:       0
    weak_bind_off:   0
    weak_bind_size:  0
    lazy_bind_off:   0
    lazy_bind_size:  0
    export_off:      32768
    export_size:     56
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          32832
    nsyms:           5
    stroff:          32912
    strsize:         24
  - cmd:             LC_DYSYMTAB
    cmdsize:         80
    ilocalsym:       0
    nlocalsym:       1
    iextdefsym:      1
    nextdefsym:      4
    iundefsym:       5
    nundefsym:       0
    tocoff:          0
    ntoc:            0
    modtaboff:       0
    nmodtab:         0
    extrefsymoff:    0
    nextrefsyms:     0
    indirectsymoff:  0
    nindirectsyms:   0
    extreloff:       0
    nextrel:         0
    locreloff:       0
    nlocrel:         0
  - cmd:             LC_UUID
    cmdsize:         24
    uuid:            9927BB37-87F9-3AAA-B3C2-98C5C531CBD5
  - cmd:             LC_BUILD_VERSION
    cmdsize:         32
    platform:        1
    minos:           720896
    sdk:             1769472
    ntools:          1
    Tools:
      - tool:            3
        version:         1771897088
  - cmd:             LC_SOURCE_VERSION
    cmdsize:         16
    version:         0
  - cmd:             LC_LOAD_DYLIB
    cmdsize:         56
    dylib:
      name:            24
      timestamp:       2
      current_version: 89063424
      compatibility_version: 65536
    Content:         '/usr/lib/libSystem.B.dylib'
    ZeroPadBytes:    6
  - cmd:             LC_FUNCTION_STARTS
    cmdsize:         16
    dataoff:         32824
    datasize:        8
  - cmd:             LC_DATA_IN_CODE
    cmdsize:         16
    dataoff:         32832
    datasize:        0
  - cmd:             LC_CODE_SIGNATURE
    cmdsize:         16
    dataoff:         32944
    datasize:        408
LinkEditData:
  ExportTrie:
    TerminalSize:    0
    NodeOffset:      0
    Name:            ''
    Flags:           0x0
    Address:         0x0
    Other:           0x0
    ImportName:      ''
    Children:
      - TerminalSize:    0
        NodeOffset:      30
        Name:            _
        Flags:           0x0
        Address:         0x0
        Other:           0x0
        ImportName:      ''
        Children:
          - TerminalSize:    3
            NodeOffset:      9
            Name:            bar
            Flags:           0x0
            Address:         0x394
            Other:           0x0
            ImportName:      ''
          - TerminalSize:    4
            NodeOffset:      14
            Name:            d
            Flags:           0x0
            Address:         0x4000
            Other:           0x0
            ImportName:      ''
          - TerminalSize:    3
            NodeOffset:      20
            Name:            foo
            Flags:           0x0
            Address:         0x388
            Other:           0x0
            ImportName:      ''
          - TerminalSize:    3
            NodeOffset:      25
            Name:            main
            Flags:           0x0
            Address:         0x3B4
            Other:           0x0
            ImportName:      ''
  NameList:
    - n_strx:          21
      n_type:          0xE
      n_sect:          3
      n_desc:          0
      n_value:         16388
    - n_strx:          2
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         916
    - n_strx:          7
      n_type:          0xF
      n_sect:          3
      n_desc:          0
      n_value:         16384
    - n_strx:          10
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         904
    - n_strx:          15
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         948
  StringTable:
    - ' '
    - _bar
    - _d
    - _foo
    - _main
    - _c
  FunctionStarts:  [ 0x388, 0x394, 0x3B4 ]
...
)");

  auto ExpectedFileNonZeroOffset = TestFile::fromYaml(R"(
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x100000C
  cpusubtype:      0x0
  filetype:        0x6
  ncmds:           14
  sizeofcmds:      824
  flags:           0x100085
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         232
    segname:         __TEXT
    vmaddr:          245760
    vmsize:          16384
    fileoff:         0
    filesize:        16384
    maxprot:         5
    initprot:        5
    nsects:          2
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x3C388
        size:            96
        offset:          0x388
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         28000090000540B9C0035FD6FD7BBFA9FD030091FBFFFF9728000090080140B90000080BFD7BC1A8C0035FD6FF8300D1FD7B01A9FD430091BFC31FB8F4FFFF9728000090080540B90800080BE80B00B9E00B40B9FD7B41A9FF830091C0035FD6
      - sectname:        __unwind_info
        segname:         __TEXT
        addr:            0x3C3E8
        size:            96
        offset:          0x3E8
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         010000001C000000000000001C000000000000001C00000002000000880300004000000040000000E90300000000000040000000000000000000000000000000030000000C00020014000200000000000C000001000000020000000400000000
  - cmd:             LC_SEGMENT_64
    cmdsize:         152
    segname:         __DATA
    vmaddr:          262144
    vmsize:          16384
    fileoff:         16384
    filesize:        16384
    maxprot:         3
    initprot:        3
    nsects:          1
    flags:           0
    Sections:
      - sectname:        __data
        segname:         __DATA
        addr:            0x40000
        size:            8
        offset:          0x4000
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         140000000A000000
  - cmd:             LC_SEGMENT_64
    cmdsize:         72
    segname:         __LINKEDIT
    vmaddr:          278528
    vmsize:          16384
    fileoff:         32768
    filesize:        584
    maxprot:         1
    initprot:        1
    nsects:          0
    flags:           0
  - cmd:             LC_ID_DYLIB
    cmdsize:         40
    dylib:
      name:            24
      timestamp:       1
      current_version: 0
      compatibility_version: 0
    Content:         liba.dylib
    ZeroPadBytes:    6
  - cmd:             LC_DYLD_INFO_ONLY
    cmdsize:         48
    rebase_off:      0
    rebase_size:     0
    bind_off:        0
    bind_size:       0
    weak_bind_off:   0
    weak_bind_size:  0
    lazy_bind_off:   0
    lazy_bind_size:  0
    export_off:      32768
    export_size:     56
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          32832
    nsyms:           5
    stroff:          32912
    strsize:         24
  - cmd:             LC_DYSYMTAB
    cmdsize:         80
    ilocalsym:       0
    nlocalsym:       1
    iextdefsym:      1
    nextdefsym:      4
    iundefsym:       5
    nundefsym:       0
    tocoff:          0
    ntoc:            0
    modtaboff:       0
    nmodtab:         0
    extrefsymoff:    0
    nextrefsyms:     0
    indirectsymoff:  0
    nindirectsyms:   0
    extreloff:       0
    nextrel:         0
    locreloff:       0
    nlocrel:         0
  - cmd:             LC_UUID
    cmdsize:         24
    uuid:            9927BB37-87F9-3AAA-B3C2-98C5C531CBD5
  - cmd:             LC_BUILD_VERSION
    cmdsize:         32
    platform:        1
    minos:           720896
    sdk:             1769472
    ntools:          1
    Tools:
      - tool:            3
        version:         1771897088
  - cmd:             LC_SOURCE_VERSION
    cmdsize:         16
    version:         0
  - cmd:             LC_LOAD_DYLIB
    cmdsize:         56
    dylib:
      name:            24
      timestamp:       2
      current_version: 89063424
      compatibility_version: 65536
    Content:         '/usr/lib/libSystem.B.dylib'
    ZeroPadBytes:    6
  - cmd:             LC_FUNCTION_STARTS
    cmdsize:         16
    dataoff:         32824
    datasize:        8
  - cmd:             LC_DATA_IN_CODE
    cmdsize:         16
    dataoff:         32832
    datasize:        0
  - cmd:             LC_CODE_SIGNATURE
    cmdsize:         16
    dataoff:         32944
    datasize:        408
LinkEditData:
  ExportTrie:
    TerminalSize:    0
    NodeOffset:      0
    Name:            ''
    Flags:           0x0
    Address:         0x0
    Other:           0x0
    ImportName:      ''
    Children:
      - TerminalSize:    0
        NodeOffset:      30
        Name:            _
        Flags:           0x0
        Address:         0x0
        Other:           0x0
        ImportName:      ''
        Children:
          - TerminalSize:    3
            NodeOffset:      9
            Name:            bar
            Flags:           0x0
            Address:         0x394
            Other:           0x0
            ImportName:      ''
          - TerminalSize:    4
            NodeOffset:      14
            Name:            d
            Flags:           0x0
            Address:         0x4000
            Other:           0x0
            ImportName:      ''
          - TerminalSize:    3
            NodeOffset:      20
            Name:            foo
            Flags:           0x0
            Address:         0x388
            Other:           0x0
            ImportName:      ''
          - TerminalSize:    3
            NodeOffset:      25
            Name:            main
            Flags:           0x0
            Address:         0x3B4
            Other:           0x0
            ImportName:      ''
  NameList:
    - n_strx:          21
      n_type:          0xE
      n_sect:          3
      n_desc:          0
      n_value:         262148
    - n_strx:          2
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         246676
    - n_strx:          7
      n_type:          0xF
      n_sect:          3
      n_desc:          0
      n_value:         262144
    - n_strx:          10
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         246664
    - n_strx:          15
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         246708
  StringTable:
    - ' '
    - _bar
    - _d
    - _foo
    - _main
    - _c
  FunctionStarts:  [ 0x388, 0x394, 0x3B4 ]
...
)");

  //
  // Create a binary for the 0-address YAML, serialize it
  // to a buffer.
  //
  ASSERT_THAT_EXPECTED(ExpectedFileZeroOffset, llvm::Succeeded());
  auto zero_module_sp =
      std::make_shared<Module>(ExpectedFileZeroOffset->moduleSpec());
  ObjectFile *zero_objfile = zero_module_sp->GetObjectFile();
  ASSERT_NE(zero_objfile, nullptr);

  Symtab *zero_module_symtab = zero_module_sp->GetSymtab();
  zero_module_symtab->PreloadSymbols();
  const uint8_t addr_size = 8;

  DataEncoder zero_file(eByteOrderLittle, addr_size);
  zero_module_symtab->Encode(zero_file);
  llvm::ArrayRef<uint8_t> zero_filebytes = zero_file.GetData();

  //
  // Create a binary for the non-0-address YAML, serialize it
  // to a buffer.
  //
  ASSERT_THAT_EXPECTED(ExpectedFileNonZeroOffset, llvm::Succeeded());
  auto nonzero_module_sp =
      std::make_shared<Module>(ExpectedFileNonZeroOffset->moduleSpec());
  ObjectFile *nonzero_objfile = nonzero_module_sp->GetObjectFile();
  ASSERT_NE(nonzero_objfile, nullptr);

  Symtab *nonzero_module_symtab = nonzero_module_sp->GetSymtab();
  ASSERT_NE(nonzero_module_symtab, nullptr);
  nonzero_module_symtab->PreloadSymbols();

  DataEncoder nonzero_file(eByteOrderLittle, addr_size);
  nonzero_module_symtab->Encode(nonzero_file);
  llvm::ArrayRef<uint8_t> nonzero_filebytes = nonzero_file.GetData();

  //
  // Compare the two serialized buffers, they must be identical.
  //
  ASSERT_EQ(zero_filebytes, nonzero_filebytes);

  //
  // Test that the symtabs round-trip through serialization and yield
  // the same Symtab as the original binary.
  //
  EncodeDecode(*zero_module_symtab);
  EncodeDecode(*nonzero_module_symtab);
}

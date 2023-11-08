//===-- SymbolTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
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
                TypeSystemClang>
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
    0x01, 0x10, 0x3e, 0x94, 0x86, 0x6e, 0x0d, 0x1a,
    0x39, 0xbd, 0x97, 0x5b, 0x64, 0xe8, 0xf1, 0xfd,
    0xba, 0xae, 0xff, 0x53, 0x54, 0x41, 0x42, 0x91,
    0x00, 0x00, 0x00, 0x00, 0x2f, 0x55, 0x73, 0x65,
    0x72, 0x73, 0x2f, 0x67, 0x63, 0x6c, 0x61, 0x79,
    0x74, 0x6f, 0x6e, 0x2f, 0x44, 0x6f, 0x63, 0x75,
    0x6d, 0x65, 0x6e, 0x74, 0x73, 0x2f, 0x6f, 0x62,
    0x6a, 0x66, 0x69, 0x6c, 0x65, 0x73, 0x2f, 0x69,
    0x6e, 0x64, 0x65, 0x78, 0x2d, 0x6e, 0x61, 0x6d,
    0x65, 0x73, 0x2f, 0x6d, 0x61, 0x69, 0x6e, 0x2e,
    0x63, 0x00, 0x2f, 0x55, 0x73, 0x65, 0x72, 0x73,
    0x2f, 0x67, 0x63, 0x6c, 0x61, 0x79, 0x74, 0x6f,
    0x6e, 0x2f, 0x44, 0x6f, 0x63, 0x75, 0x6d, 0x65,
    0x6e, 0x74, 0x73, 0x2f, 0x6f, 0x62, 0x6a, 0x66,
    0x69, 0x6c, 0x65, 0x73, 0x2f, 0x69, 0x6e, 0x64,
    0x65, 0x78, 0x2d, 0x6e, 0x61, 0x6d, 0x65, 0x73,
    0x2f, 0x6d, 0x61, 0x69, 0x6e, 0x2e, 0x6f, 0x00,
    0x66, 0x6f, 0x6f, 0x00, 0x62, 0x61, 0x72, 0x00,
    0x62, 0x61, 0x7a, 0x00, 0x6d, 0x61, 0x69, 0x6e,
    0x00, 0x5f, 0x6d, 0x68, 0x5f, 0x65, 0x78, 0x65,
    0x63, 0x75, 0x74, 0x65, 0x5f, 0x68, 0x65, 0x61,
    0x64, 0x65, 0x72, 0x00, 0x53, 0x59, 0x4d, 0x42,
    0x01, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x2a,
    0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x64, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0a, 0x20, 0x01, 0x37, 0x00, 0x00, 0x00, 0x00,
    0x7b, 0xc3, 0x69, 0x62, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x66, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x02, 0x32, 0x01, 0x6d, 0x00, 0x00,
    0x00, 0x01, 0x64, 0x3f, 0x00, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x08, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x32, 0x01, 0x71,
    0x00, 0x00, 0x00, 0x01, 0x6c, 0x3f, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x32,
    0x01, 0x75, 0x00, 0x00, 0x00, 0x01, 0x74, 0x3f,
    0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0f, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x32, 0x01, 0x79, 0x00, 0x00, 0x00, 0x01,
    0x7c, 0x3f, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x0f, 0x00, 0x14, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x04, 0x12, 0x02, 0x7e, 0x00, 0x00,
    0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x64, 0x3f, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x10, 0x00, 0x0f, 0x00, 0x01, 0x00,
    0x43, 0x4d, 0x41, 0x50, 0x07, 0x00, 0x00, 0x00,
    0x6d, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x75, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x79, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
    0x37, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x71, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x7e, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00
  };

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
  Symbol *symbol = nullptr;
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

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

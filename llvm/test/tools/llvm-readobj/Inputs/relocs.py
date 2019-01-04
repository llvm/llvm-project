#!/usr/bin/env python

from __future__ import print_function

# Generates ELF, COFF and MachO object files for different architectures
# containing all relocations:
#
# ELF:   i386, x86_64, ppc64, aarch64, arm, mips, mips64el
# COFF:  i386, x86_64
# MachO: i386, x86_64, arm
# (see end of file for triples)
#
# To simplify generation, object files are generated with just the proper
# number of relocations through repeated instructions. Afterwards, the
# relocations in the object file are patched to their proper value.

import operator
import shutil
import struct
import subprocess
import sys

class EnumType(type):
  def __init__(self, name, bases = (), attributes = {}):
    super(EnumType, self).__init__(name, bases, attributes)

    type.__setattr__(self, '_map', {})
    type.__setattr__(self, '_nameMap', {})

    for symbol in attributes:
      if symbol.startswith('__') or symbol.endswith('__'):
        continue

      value = attributes[symbol]

      # MyEnum.symbol == value
      type.__setattr__(self, symbol, value)
      self._nameMap[symbol] = value

      # The first symbol with the given value is authoritative.
      if not (value in self._map):
        # MyEnum[value] == symbol
        self._map[value] = symbol

  # Not supported (Enums are immutable).
  def __setattr__(self, name, value):
    raise NotSupportedException(self.__setattr__)

  # Not supported (Enums are immutable).
  def __delattr__(self, name):
    raise NotSupportedException(self.__delattr__)

  # Gets the enum symbol for the specified value.
  def __getitem__(self, value):
    symbol = self._map.get(value)
    if symbol is None:
      raise KeyError(value)
    return symbol

  # Gets the enum symbol for the specified value or none.
  def lookup(self, value):
    symbol = self._map.get(value)
    return symbol

  # Not supported (Enums are immutable).
  def __setitem__(self, value, symbol):
    raise NotSupportedException(self.__setitem__)

  # Not supported (Enums are immutable).
  def __delitem__(self, value):
    raise NotSupportedException(self.__delitem__)

  def entries(self):
    # sort by (value, name)
    def makeKey(item):
      return (item[1], item[0])
    e = []
    for pair in sorted(self._nameMap.items(), key=makeKey):
      e.append(pair)
    return e

  def __iter__(self):
    for e in self.entries():
      yield e

Enum = EnumType('Enum', (), {})

class BinaryReader:
  def __init__(self, path):
    self.file = open(path, "r+b", 0)
    self.isLSB = None
    self.is64Bit = None
    self.isN64 = False

  def tell(self):
    return self.file.tell()

  def seek(self, pos):
    self.file.seek(pos)

  def read(self, N):
    data = self.file.read(N)
    if len(data) != N:
      raise ValueError("Out of data!")
    return data

  def int8(self):
    return ord(self.read(1))

  def uint8(self):
    return ord(self.read(1))

  def int16(self):
    return struct.unpack('><'[self.isLSB] + 'h', self.read(2))[0]

  def uint16(self):
    return struct.unpack('><'[self.isLSB] + 'H', self.read(2))[0]

  def int32(self):
    return struct.unpack('><'[self.isLSB] + 'i', self.read(4))[0]

  def uint32(self):
    return struct.unpack('><'[self.isLSB] + 'I', self.read(4))[0]

  def int64(self):
    return struct.unpack('><'[self.isLSB] + 'q', self.read(8))[0]

  def uint64(self):
    return struct.unpack('><'[self.isLSB] + 'Q', self.read(8))[0]

  def writeUInt8(self, value):
    self.file.write(struct.pack('><'[self.isLSB] + 'B', value))

  def writeUInt16(self, value):
    self.file.write(struct.pack('><'[self.isLSB] + 'H', value))

  def writeUInt32(self, value):
    self.file.write(struct.pack('><'[self.isLSB] + 'I', value))

  def writeUInt64(self, value):
    self.file.write(struct.pack('><'[self.isLSB] + 'Q', value))

  def word(self):
    if self.is64Bit:
      return self.uint64()
    else:
      return self.uint32()

  def writeWord(self, value):
    if self.is64Bit:
      self.writeUInt64(value)
    else:
      self.writeUInt32(value)

class StringTable:
  def __init__(self, strings):
    self.string_table = strings

  def __getitem__(self, index):
    end = self.string_table.index('\x00', index)
    return self.string_table[index:end]

class ElfSection:
  def __init__(self, f):
    self.sh_name = f.uint32()
    self.sh_type = f.uint32()
    self.sh_flags = f.word()
    self.sh_addr = f.word()
    self.sh_offset = f.word()
    self.sh_size = f.word()
    self.sh_link = f.uint32()
    self.sh_info = f.uint32()
    self.sh_addralign = f.word()
    self.sh_entsize = f.word()

  def patch(self, f, relocs):
    if self.sh_type == 4 or self.sh_type == 9: # SHT_RELA / SHT_REL
      self.patchRelocs(f, relocs)

  def patchRelocs(self, f, relocs):
    entries = self.sh_size // self.sh_entsize

    for index in range(entries):
      f.seek(self.sh_offset + index * self.sh_entsize)
      r_offset = f.word()

      if index < len(relocs):
        ri = index
      else:
        ri = 0

      if f.isN64:
        r_sym =   f.uint32()
        r_ssym =  f.uint8()
        f.seek(f.tell())
        f.writeUInt8(relocs[ri][1])
        f.writeUInt8(relocs[ri][1])
        f.writeUInt8(relocs[ri][1])
      else:
        pos = f.tell()
        r_info = f.word()

        r_type = relocs[ri][1]
        if f.is64Bit:
          r_info = (r_info & 0xFFFFFFFF00000000) | (r_type & 0xFFFFFFFF)
        else:
          r_info = (r_info & 0xFF00) | (r_type & 0xFF)

        print("    %s" % relocs[ri][0])
        f.seek(pos)
        f.writeWord(r_info)


class CoffSection:
  def __init__(self, f):
    self.raw_name                = f.read(8)
    self.virtual_size            = f.uint32()
    self.virtual_address         = f.uint32()
    self.raw_data_size           = f.uint32()
    self.pointer_to_raw_data     = f.uint32()
    self.pointer_to_relocations  = f.uint32()
    self.pointer_to_line_numbers = f.uint32()
    self.relocation_count        = f.uint16()
    self.line_number_count       = f.uint16()
    self.characteristics         = f.uint32()


def compileAsm(filename, triple, src):
  cmd = ["llvm-mc", "-triple=" + triple, "-filetype=obj", "-o", filename]
  print("  Running: " + " ".join(cmd))
  p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
  p.communicate(input=src)
  p.wait()

def compileIR(filename, triple, src):
  cmd = ["llc", "-mtriple=" + triple, "-filetype=obj", "-o", filename]
  print("  Running: " + " ".join(cmd))
  p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
  p.communicate(input=src)
  p.wait()


def craftElf(filename, triple, relocs, dummyReloc):
  print("Crafting " + filename + " for " + triple)
  if type(dummyReloc) is tuple:
    preSrc, dummyReloc, relocsPerDummy = dummyReloc
    src = preSrc + "\n"
    for i in range((len(relocs) + relocsPerDummy - 1) / relocsPerDummy):
      src += dummyReloc.format(i) + "\n"
    compileIR(filename, triple, src)
  else:
    src = (dummyReloc + "\n") * len(relocs)
    compileAsm(filename, triple, src)

  print("  Patching relocations...")
  patchElf(filename, relocs)

def patchElf(path, relocs):
  f = BinaryReader(path)

  magic = f.read(4)
  assert magic == '\x7FELF'

  fileclass = f.uint8()
  if fileclass == 1:
    f.is64Bit = False
  elif fileclass == 2:
    f.is64Bit = True
  else:
    raise ValueError("Unknown file class %x" % fileclass)

  byteordering = f.uint8()
  if byteordering == 1:
      f.isLSB = True
  elif byteordering == 2:
      f.isLSB = False
  else:
      raise ValueError("Unknown byte ordering %x" % byteordering)

  f.seek(18)
  e_machine = f.uint16()
  if e_machine == 0x0008 and f.is64Bit: # EM_MIPS && 64 bit
      f.isN64 = True

  e_version = f.uint32()
  e_entry = f.word()
  e_phoff = f.word()
  e_shoff = f.word()
  e_flags = f.uint32()
  e_ehsize = f.uint16()
  e_phentsize = f.uint16()
  e_phnum = f.uint16()
  e_shentsize = f.uint16()
  e_shnum = f.uint16()
  e_shstrndx = f.uint16()

  sections = []
  for index in range(e_shnum):
    f.seek(e_shoff + index * e_shentsize)
    s = ElfSection(f)
    sections.append(s)

  f.seek(sections[e_shstrndx].sh_offset)
  shstrtab = StringTable(f.read(sections[e_shstrndx].sh_size))

  strtab = None
  for section in sections:
    if shstrtab[section.sh_name] == ".strtab":
      f.seek(section.sh_offset)
      strtab = StringTable(f.read(section.sh_size))
      break

  for index in range(e_shnum):
    sections[index].patch(f, relocs)


def craftCoff(filename, triple, relocs, dummyReloc):
  print("Crafting " + filename + " for " + triple)
  src = (dummyReloc + "\n") * len(relocs)
  compileAsm(filename, triple, src)

  print("  Patching relocations...")
  patchCoff(filename, relocs)

def patchCoff(path, relocs):
  f = BinaryReader(path)
  f.isLSB = True

  machine_type            = f.uint16()
  section_count           = f.uint16()

  # Zero out timestamp to prevent churn when regenerating COFF files.
  f.writeUInt32(0)

  f.seek(20)
  sections = [CoffSection(f) for idx in range(section_count)]

  section = sections[0]
  f.seek(section.pointer_to_relocations)
  for i in range(section.relocation_count):
    virtual_addr = f.uint32()
    symtab_idx   = f.uint32()
    print("    %s" % relocs[i][0])
    f.writeUInt16(relocs[i][1])


def craftMacho(filename, triple, relocs, dummyReloc):
  print("Crafting " + filename + " for " + triple)

  if type(dummyReloc) is tuple:
    srcType, preSrc, dummyReloc, relocsPerDummy = dummyReloc
    src = preSrc + "\n"
    for i in range((len(relocs) + relocsPerDummy - 1) / relocsPerDummy):
      src += dummyReloc.format(i) + "\n"
    if srcType == "asm":
      compileAsm(filename, triple, src)
    elif srcType == "ir":
      compileIR(filename, triple, src)
  else:
    src = (dummyReloc + "\n") * len(relocs)
    compileAsm(filename, triple, src)

  print("  Patching relocations...")
  patchMacho(filename, relocs)

def patchMacho(filename, relocs):
  f = BinaryReader(filename)

  magic = f.read(4)
  if magic == '\xFE\xED\xFA\xCE':
    f.isLSB, f.is64Bit = False, False
  elif magic == '\xCE\xFA\xED\xFE':
    f.isLSB, f.is64Bit = True, False
  elif magic == '\xFE\xED\xFA\xCF':
    f.isLSB, f.is64Bit = False, True
  elif magic == '\xCF\xFA\xED\xFE':
    f.isLSB, f.is64Bit = True, True
  else:
    raise ValueError("Not a Mach-O object file: %r (bad magic)" % path)

  cputype = f.uint32()
  cpusubtype = f.uint32()
  filetype = f.uint32()
  numLoadCommands = f.uint32()
  loadCommandsSize = f.uint32()
  flag = f.uint32()
  if f.is64Bit:
    reserved = f.uint32()

  start = f.tell()

  for i in range(numLoadCommands):
    patchMachoLoadCommand(f, relocs)

  if f.tell() - start != loadCommandsSize:
    raise ValueError("%s: warning: invalid load commands size: %r" % (
      sys.argv[0], loadCommandsSize))

def patchMachoLoadCommand(f, relocs):
  start = f.tell()
  cmd = f.uint32()
  cmdSize = f.uint32()

  if cmd == 1:
    patchMachoSegmentLoadCommand(f, relocs)
  elif cmd == 25:
    patchMachoSegmentLoadCommand(f, relocs)
  else:
    f.read(cmdSize - 8)

  if f.tell() - start != cmdSize:
    raise ValueError("%s: warning: invalid load command size: %r" % (
      sys.argv[0], cmdSize))

def patchMachoSegmentLoadCommand(f, relocs):
  segment_name = f.read(16)
  vm_addr = f.word()
  vm_size = f.word()
  file_offset = f.word()
  file_size = f.word()
  maxprot = f.uint32()
  initprot = f.uint32()
  numSections = f.uint32()
  flags = f.uint32()
  for i in range(numSections):
    patchMachoSection(f, relocs)

def patchMachoSection(f, relocs):
  section_name = f.read(16)
  segment_name = f.read(16)
  address = f.word()
  size = f.word()
  offset = f.uint32()
  alignment = f.uint32()
  relocOffset = f.uint32()
  numReloc = f.uint32()
  flags = f.uint32()
  reserved1 = f.uint32()
  reserved2 = f.uint32()
  if f.is64Bit:
    reserved3 = f.uint32()

  prev_pos = f.tell()

  f.seek(relocOffset)
  for i in range(numReloc):
    ri = i < len(relocs) and i or 0
    print("    %s" % relocs[ri][0])
    word1 = f.uint32()
    pos = f.tell()
    value = f.uint32()
    f.seek(pos)
    value = (value & 0x0FFFFFFF) | ((relocs[ri][1] & 0xF) << 28)
    f.writeUInt32(value)
  f.seek(prev_pos)


class Relocs_Elf_X86_64(Enum):
  R_X86_64_NONE       = 0
  R_X86_64_64         = 1
  R_X86_64_PC32       = 2
  R_X86_64_GOT32      = 3
  R_X86_64_PLT32      = 4
  R_X86_64_COPY       = 5
  R_X86_64_GLOB_DAT   = 6
  R_X86_64_JUMP_SLOT  = 7
  R_X86_64_RELATIVE   = 8
  R_X86_64_GOTPCREL   = 9
  R_X86_64_32         = 10
  R_X86_64_32S        = 11
  R_X86_64_16         = 12
  R_X86_64_PC16       = 13
  R_X86_64_8          = 14
  R_X86_64_PC8        = 15
  R_X86_64_DTPMOD64   = 16
  R_X86_64_DTPOFF64   = 17
  R_X86_64_TPOFF64    = 18
  R_X86_64_TLSGD      = 19
  R_X86_64_TLSLD      = 20
  R_X86_64_DTPOFF32   = 21
  R_X86_64_GOTTPOFF   = 22
  R_X86_64_TPOFF32    = 23
  R_X86_64_PC64       = 24
  R_X86_64_GOTOFF64   = 25
  R_X86_64_GOTPC32    = 26
  R_X86_64_GOT64      = 27
  R_X86_64_GOTPCREL64 = 28
  R_X86_64_GOTPC64    = 29
  R_X86_64_GOTPLT64   = 30
  R_X86_64_PLTOFF64   = 31
  R_X86_64_SIZE32     = 32
  R_X86_64_SIZE64     = 33
  R_X86_64_GOTPC32_TLSDESC = 34
  R_X86_64_TLSDESC_CALL    = 35
  R_X86_64_TLSDESC    = 36
  R_X86_64_IRELATIVE  = 37

class Relocs_Elf_i386(Enum):
  R_386_NONE          = 0
  R_386_32            = 1
  R_386_PC32          = 2
  R_386_GOT32         = 3
  R_386_PLT32         = 4
  R_386_COPY          = 5
  R_386_GLOB_DAT      = 6
  R_386_JUMP_SLOT     = 7
  R_386_RELATIVE      = 8
  R_386_GOTOFF        = 9
  R_386_GOTPC         = 10
  R_386_32PLT         = 11
  R_386_TLS_TPOFF     = 14
  R_386_TLS_IE        = 15
  R_386_TLS_GOTIE     = 16
  R_386_TLS_LE        = 17
  R_386_TLS_GD        = 18
  R_386_TLS_LDM       = 19
  R_386_16            = 20
  R_386_PC16          = 21
  R_386_8             = 22
  R_386_PC8           = 23
  R_386_TLS_GD_32     = 24
  R_386_TLS_GD_PUSH   = 25
  R_386_TLS_GD_CALL   = 26
  R_386_TLS_GD_POP    = 27
  R_386_TLS_LDM_32    = 28
  R_386_TLS_LDM_PUSH  = 29
  R_386_TLS_LDM_CALL  = 30
  R_386_TLS_LDM_POP   = 31
  R_386_TLS_LDO_32    = 32
  R_386_TLS_IE_32     = 33
  R_386_TLS_LE_32     = 34
  R_386_TLS_DTPMOD32  = 35
  R_386_TLS_DTPOFF32  = 36
  R_386_TLS_TPOFF32   = 37
  R_386_TLS_GOTDESC   = 39
  R_386_TLS_DESC_CALL = 40
  R_386_TLS_DESC      = 41
  R_386_IRELATIVE     = 42
  R_386_NUM           = 43

class Relocs_Elf_PPC32(Enum):
  R_PPC_NONE                  = 0
  R_PPC_ADDR32                = 1
  R_PPC_ADDR24                = 2
  R_PPC_ADDR16                = 3
  R_PPC_ADDR16_LO             = 4
  R_PPC_ADDR16_HI             = 5
  R_PPC_ADDR16_HA             = 6
  R_PPC_ADDR14                = 7
  R_PPC_ADDR14_BRTAKEN        = 8
  R_PPC_ADDR14_BRNTAKEN       = 9
  R_PPC_REL24                 = 10
  R_PPC_REL14                 = 11
  R_PPC_REL14_BRTAKEN         = 12
  R_PPC_REL14_BRNTAKEN        = 13
  R_PPC_REL32                 = 26
  R_PPC_TPREL16_LO            = 70
  R_PPC_TPREL16_HA            = 72

class Relocs_Elf_PPC64(Enum):
  R_PPC64_NONE                = 0
  R_PPC64_ADDR32              = 1
  R_PPC64_ADDR16_LO           = 4
  R_PPC64_ADDR16_HI           = 5
  R_PPC64_ADDR14              = 7
  R_PPC64_REL24               = 10
  R_PPC64_REL32               = 26
  R_PPC64_ADDR64              = 38
  R_PPC64_ADDR16_HIGHER       = 39
  R_PPC64_ADDR16_HIGHEST      = 41
  R_PPC64_REL64               = 44
  R_PPC64_TOC16               = 47
  R_PPC64_TOC16_LO            = 48
  R_PPC64_TOC16_HA            = 50
  R_PPC64_TOC                 = 51
  R_PPC64_ADDR16_DS           = 56
  R_PPC64_ADDR16_LO_DS        = 57
  R_PPC64_TOC16_DS            = 63
  R_PPC64_TOC16_LO_DS         = 64
  R_PPC64_TLS                 = 67
  R_PPC64_TPREL16_LO          = 70
  R_PPC64_TPREL16_HA          = 72
  R_PPC64_DTPREL16_LO         = 75
  R_PPC64_DTPREL16_HA         = 77
  R_PPC64_GOT_TLSGD16_LO      = 80
  R_PPC64_GOT_TLSGD16_HA      = 82
  R_PPC64_GOT_TLSLD16_LO      = 84
  R_PPC64_GOT_TLSLD16_HA      = 86
  R_PPC64_GOT_TPREL16_LO_DS   = 88
  R_PPC64_GOT_TPREL16_HA      = 90
  R_PPC64_TLSGD               = 107
  R_PPC64_TLSLD               = 108

class Relocs_Elf_AArch64(Enum):
  R_AARCH64_NONE                        = 0
  R_AARCH64_ABS64                       = 0x101
  R_AARCH64_ABS32                       = 0x102
  R_AARCH64_ABS16                       = 0x103
  R_AARCH64_PREL64                      = 0x104
  R_AARCH64_PREL32                      = 0x105
  R_AARCH64_PREL16                      = 0x106
  R_AARCH64_MOVW_UABS_G0                = 0x107
  R_AARCH64_MOVW_UABS_G0_NC             = 0x108
  R_AARCH64_MOVW_UABS_G1                = 0x109
  R_AARCH64_MOVW_UABS_G1_NC             = 0x10a
  R_AARCH64_MOVW_UABS_G2                = 0x10b
  R_AARCH64_MOVW_UABS_G2_NC             = 0x10c
  R_AARCH64_MOVW_UABS_G3                = 0x10d
  R_AARCH64_MOVW_SABS_G0                = 0x10e
  R_AARCH64_MOVW_SABS_G1                = 0x10f
  R_AARCH64_MOVW_SABS_G2                = 0x110
  R_AARCH64_LD_PREL_LO19                = 0x111
  R_AARCH64_ADR_PREL_LO21               = 0x112
  R_AARCH64_ADR_PREL_PG_HI21            = 0x113
  R_AARCH64_ADR_PREL_PG_HI21_NC         = 0x114
  R_AARCH64_ADD_ABS_LO12_NC             = 0x115
  R_AARCH64_LDST8_ABS_LO12_NC           = 0x116
  R_AARCH64_TSTBR14                     = 0x117
  R_AARCH64_CONDBR19                    = 0x118
  R_AARCH64_JUMP26                      = 0x11a
  R_AARCH64_CALL26                      = 0x11b
  R_AARCH64_LDST16_ABS_LO12_NC          = 0x11c
  R_AARCH64_LDST32_ABS_LO12_NC          = 0x11d
  R_AARCH64_LDST64_ABS_LO12_NC          = 0x11e
  R_AARCH64_MOVW_PREL_G0                = 0x11f
  R_AARCH64_MOVW_PREL_G0_NC             = 0x120
  R_AARCH64_MOVW_PREL_G1                = 0x121
  R_AARCH64_MOVW_PREL_G1_NC             = 0x122
  R_AARCH64_MOVW_PREL_G2                = 0x123
  R_AARCH64_MOVW_PREL_G2_NC             = 0x124
  R_AARCH64_MOVW_PREL_G3                = 0x125
  R_AARCH64_LDST128_ABS_LO12_NC         = 0x12b
  R_AARCH64_MOVW_GOTOFF_G0              = 0x12c
  R_AARCH64_MOVW_GOTOFF_G0_NC           = 0x12d
  R_AARCH64_MOVW_GOTOFF_G1              = 0x12e
  R_AARCH64_MOVW_GOTOFF_G1_NC           = 0x12f
  R_AARCH64_MOVW_GOTOFF_G2              = 0x130
  R_AARCH64_MOVW_GOTOFF_G2_NC           = 0x131
  R_AARCH64_MOVW_GOTOFF_G3              = 0x132
  R_AARCH64_GOTREL64                    = 0x133
  R_AARCH64_GOTREL32                    = 0x134
  R_AARCH64_GOT_LD_PREL19               = 0x135
  R_AARCH64_LD64_GOTOFF_LO15            = 0x136
  R_AARCH64_ADR_GOT_PAGE                = 0x137
  R_AARCH64_LD64_GOT_LO12_NC            = 0x138
  R_AARCH64_LD64_GOTPAGE_LO15           = 0x139
  R_AARCH64_TLSGD_ADR_PREL21            = 0x200
  R_AARCH64_TLSGD_ADR_PAGE21            = 0x201
  R_AARCH64_TLSGD_ADD_LO12_NC           = 0x202
  R_AARCH64_TLSGD_MOVW_G1               = 0x203
  R_AARCH64_TLSGD_MOVW_G0_NC            = 0x204
  R_AARCH64_TLSLD_ADR_PREL21            = 0x205
  R_AARCH64_TLSLD_ADR_PAGE21            = 0x206
  R_AARCH64_TLSLD_ADD_LO12_NC           = 0x207
  R_AARCH64_TLSLD_MOVW_G1               = 0x208
  R_AARCH64_TLSLD_MOVW_G0_NC            = 0x209
  R_AARCH64_TLSLD_LD_PREL19             = 0x20a
  R_AARCH64_TLSLD_MOVW_DTPREL_G2        = 0x20b
  R_AARCH64_TLSLD_MOVW_DTPREL_G1        = 0x20c
  R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC     = 0x20d
  R_AARCH64_TLSLD_MOVW_DTPREL_G0        = 0x20e
  R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC     = 0x20f
  R_AARCH64_TLSLD_ADD_DTPREL_HI12       = 0x210
  R_AARCH64_TLSLD_ADD_DTPREL_LO12       = 0x211
  R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC    = 0x212
  R_AARCH64_TLSLD_LDST8_DTPREL_LO12     = 0x213
  R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC  = 0x214
  R_AARCH64_TLSLD_LDST16_DTPREL_LO12    = 0x215
  R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC = 0x216
  R_AARCH64_TLSLD_LDST32_DTPREL_LO12    = 0x217
  R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC = 0x218
  R_AARCH64_TLSLD_LDST64_DTPREL_LO12    = 0x219
  R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC = 0x21a
  R_AARCH64_TLSIE_MOVW_GOTTPREL_G1      = 0x21b
  R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC   = 0x21c
  R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21   = 0x21d
  R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC = 0x21e
  R_AARCH64_TLSIE_LD_GOTTPREL_PREL19    = 0x21f
  R_AARCH64_TLSLE_MOVW_TPREL_G2         = 0x220
  R_AARCH64_TLSLE_MOVW_TPREL_G1         = 0x221
  R_AARCH64_TLSLE_MOVW_TPREL_G1_NC      = 0x222
  R_AARCH64_TLSLE_MOVW_TPREL_G0         = 0x223
  R_AARCH64_TLSLE_MOVW_TPREL_G0_NC      = 0x224
  R_AARCH64_TLSLE_ADD_TPREL_HI12        = 0x225
  R_AARCH64_TLSLE_ADD_TPREL_LO12        = 0x226
  R_AARCH64_TLSLE_ADD_TPREL_LO12_NC     = 0x227
  R_AARCH64_TLSLE_LDST8_TPREL_LO12      = 0x228
  R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC   = 0x229
  R_AARCH64_TLSLE_LDST16_TPREL_LO12     = 0x22a
  R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC  = 0x22b
  R_AARCH64_TLSLE_LDST32_TPREL_LO12     = 0x22c
  R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC  = 0x22d
  R_AARCH64_TLSLE_LDST64_TPREL_LO12     = 0x22e
  R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC  = 0x22f
  R_AARCH64_TLSDESC_LD_PREL19           = 0x230
  R_AARCH64_TLSDESC_ADR_PREL21          = 0x231
  R_AARCH64_TLSDESC_ADR_PAGE21          = 0x232
  R_AARCH64_TLSDESC_LD64_LO12_NC        = 0x233
  R_AARCH64_TLSDESC_ADD_LO12_NC         = 0x234
  R_AARCH64_TLSDESC_OFF_G1              = 0x235
  R_AARCH64_TLSDESC_OFF_G0_NC           = 0x236
  R_AARCH64_TLSDESC_LDR                 = 0x237
  R_AARCH64_TLSDESC_ADD                 = 0x238
  R_AARCH64_TLSDESC_CALL                = 0x239
  R_AARCH64_TLSLE_LDST128_TPREL_LO12    = 0x23a
  R_AARCH64_TLSLE_LDST128_TPREL_LO12_NC = 0x23b
  R_AARCH64_TLSLD_LDST128_DTPREL_LO12   = 0x23c
  R_AARCH64_TLSLD_LDST128_DTPREL_LO12_NC = 0x23d
  R_AARCH64_COPY                        = 0x400
  R_AARCH64_GLOB_DAT                    = 0x401
  R_AARCH64_JUMP_SLOT                   = 0x402
  R_AARCH64_RELATIVE                    = 0x403
  R_AARCH64_TLS_DTPREL64                = 0x404
  R_AARCH64_TLS_DTPMOD64                = 0x405
  R_AARCH64_TLS_TPREL64                 = 0x406
  R_AARCH64_TLSDESC                     = 0x407
  R_AARCH64_IRELATIVE                   = 0x408

class Relocs_Elf_AArch64_ILP32(Enum):
  R_AARCH64_P32_NONE                         = 0
  R_AARCH64_P32_ABS32                        = 1
  R_AARCH64_P32_ABS16                        = 2
  R_AARCH64_P32_PREL32                       = 3
  R_AARCH64_P32_PREL16                       = 4
  R_AARCH64_P32_MOVW_UABS_G0                 = 5
  R_AARCH64_P32_MOVW_UABS_G0_NC              = 6
  R_AARCH64_P32_MOVW_UABS_G1                 = 7
  R_AARCH64_P32_MOVW_SABS_G0                 = 8
  R_AARCH64_P32_LD_PREL_LO19                 = 9
  R_AARCH64_P32_ADR_PREL_LO21                = 10
  R_AARCH64_P32_ADR_PREL_PG_HI21             = 11
  R_AARCH64_P32_ADD_ABS_LO12_NC              = 12
  R_AARCH64_P32_LDST8_ABS_LO12_NC            = 13
  R_AARCH64_P32_LDST16_ABS_LO12_NC           = 14
  R_AARCH64_P32_LDST32_ABS_LO12_NC           = 15
  R_AARCH64_P32_LDST64_ABS_LO12_NC           = 16
  R_AARCH64_P32_LDST128_ABS_LO12_NC          = 17
  R_AARCH64_P32_TSTBR14                      = 18
  R_AARCH64_P32_CONDBR19                     = 19
  R_AARCH64_P32_JUMP26                       = 20
  R_AARCH64_P32_CALL26                       = 21
  R_AARCH64_P32_MOVW_PREL_G0                 = 22
  R_AARCH64_P32_MOVW_PREL_G0_NC              = 23
  R_AARCH64_P32_MOVW_PREL_G1                 = 24
  R_AARCH64_P32_GOT_LD_PREL19                = 25
  R_AARCH64_P32_ADR_GOT_PAGE                 = 26
  R_AARCH64_P32_LD32_GOT_LO12_NC             = 27
  R_AARCH64_P32_LD32_GOTPAGE_LO14            = 28
  R_AARCH64_P32_TLSGD_ADR_PREL21             = 80
  R_AARCH64_P32_TLS_GD_ADR_PAGE21            = 81
  R_AARCH64_P32_TLSGD_ADD_LO12_NC            = 82
  R_AARCH64_P32_TLSLD_ADR_PREL21             = 83
  R_AARCH64_P32_TLDLD_ADR_PAGE21             = 84
  R_AARCH64_P32_TLSLD_ADR_LO12_NC            = 85
  R_AARCH64_P32_TLSLD_LD_PREL19              = 86
  R_AARCH64_P32_TLDLD_MOVW_DTPREL_G1         = 87
  R_AARCH64_P32_TLSLD_MOVW_DTPREL_G0         = 88
  R_AARCH64_P32_TLSLD_MOVW_DTPREL_G0_NC      = 89
  R_AARCH64_P32_TLSLD_MOVW_ADD_DTPREL_HI12   = 90
  R_AARCH64_P32_TLSLD_ADD_DTPREL_LO12        = 91
  R_AARCH64_P32_TLSLD_ADD_DTPREL_LO12_NC     = 92
  R_AARCH64_P32_TLSLD_LDST8_DTPREL_LO12      = 93
  R_AARCH64_P32_TLSLD_LDST8_DTPREL_LO12_NC   = 94
  R_AARCH64_P32_TLSLD_LDST16_DTPREL_LO12     = 95
  R_AARCH64_P32_TLSLD_LDST16_DTPREL_LO12_NC  = 96
  R_AARCH64_P32_TLSLD_LDST32_DTPREL_LO12     = 97
  R_AARCH64_P32_TLSLD_LDST32_DTPREL_LO12_NC  = 98
  R_AARCH64_P32_TLSLD_LDST64_DTPREL_LO12     = 99
  R_AARCH64_P32_TLSLD_LDST64_DTPREL_LO12_NC  = 100
  R_AARCH64_P32_TLSLD_LDST128_DTPREL_LO12    = 101
  R_AARCH64_P32_TLSLD_LDST128_DTPREL_LO12_NC = 102
  R_AARCH64_P32_TLSIE_MOVW_GOTTPREL_PAGE21   = 103
  R_AARCH64_P32_TLSIE_LD32_GOTTPREL_LO12_NC  = 104
  R_AARCH64_P32_TLSIE_LD_GOTTPREL_PREL19     = 105
  R_AARCH64_P32_TLSLE_MOVEW_TPREL_G1         = 106
  R_AARCH64_P32_TLSLE_MOVW_TPREL_G0          = 107
  R_AARCH64_P32_TLSLE_MOVW_TPREL_G0_NC       = 108
  R_AARCH64_P32_TLS_MOVW_TPREL_HI12          = 109
  R_AARCH64_P32_TLSLE_ADD_TPREL_LO12         = 110
  R_AARCH64_P32_TLSLE_ADD_TPREL_LO12_NC      = 111
  R_AARCH64_P32_TLSLE_LDST8_TPREL_LO12       = 112
  R_AARCH64_P32_TLSLE_LDST8_TPREL_LO12_NC    = 113
  R_AARCH64_P32_TLSLE_LDST16_TPREL_LO12      = 114
  R_AARCH64_P32_TLSLE_LDST16_TPREL_LO12_NC   = 115
  R_AARCH64_P32_TLSLE_LDST32_TPREL_LO12      = 116
  R_AARCH64_P32_TLSLE_LDST32_TPREL_LO12_NC   = 117
  R_AARCH64_P32_TLSLE_LDST64_TPREL_LO12      = 118
  R_AARCH64_P32_TLSLE_LDST64_TPREL_LO12_NC   = 119
  R_AARCH64_P32_TLSLE_LDST128_TPREL_LO12     = 120
  R_AARCH64_P32_TLSLE_LDST128_TPREL_LO12_NC  = 121
  R_AARCH64_P32_TLSDESC_LD_PRELL19           = 122
  R_AARCH64_P32_TLSDESC_ADR_PREL21           = 123
  R_AARCH64_P32_TLSDESC_ADR_PAGE21           = 124
  R_AARCH64_P32_TLSDESSC_LD32_LO12           = 125
  R_AARCH64_P32_TLSDESC_ADD_LO12             = 126
  R_AARCH64_P32_TLSDESC_CALL                 = 127
  R_AARCH64_P32_COPY                         = 180
  R_AARCH64_P32_GLOB_DAT                     = 181
  R_AARCH64_P32_JUMP_SLOT                    = 182
  R_AARCH64_P32_RELATIVE                     = 183
  R_AARCH64_P32_TLS_DTPREL                   = 184
  R_AARCH64_P32_TLS_DTPMOD                   = 185
  R_AARCH64_P32_TLS_TPREL                    = 186
  R_AARCH64_P32_TLSDESC                      = 187
  R_AARCH64_P32_IRELATIVE                    = 188

class Relocs_Elf_ARM(Enum):
  R_ARM_NONE                  = 0x00
  R_ARM_PC24                  = 0x01
  R_ARM_ABS32                 = 0x02
  R_ARM_REL32                 = 0x03
  R_ARM_LDR_PC_G0             = 0x04
  R_ARM_ABS16                 = 0x05
  R_ARM_ABS12                 = 0x06
  R_ARM_THM_ABS5              = 0x07
  R_ARM_ABS8                  = 0x08
  R_ARM_SBREL32               = 0x09
  R_ARM_THM_CALL              = 0x0a
  R_ARM_THM_PC8               = 0x0b
  R_ARM_BREL_ADJ              = 0x0c
  R_ARM_TLS_DESC              = 0x0d
  R_ARM_THM_SWI8              = 0x0e
  R_ARM_XPC25                 = 0x0f
  R_ARM_THM_XPC22             = 0x10
  R_ARM_TLS_DTPMOD32          = 0x11
  R_ARM_TLS_DTPOFF32          = 0x12
  R_ARM_TLS_TPOFF32           = 0x13
  R_ARM_COPY                  = 0x14
  R_ARM_GLOB_DAT              = 0x15
  R_ARM_JUMP_SLOT             = 0x16
  R_ARM_RELATIVE              = 0x17
  R_ARM_GOTOFF32              = 0x18
  R_ARM_BASE_PREL             = 0x19
  R_ARM_GOT_BREL              = 0x1a
  R_ARM_PLT32                 = 0x1b
  R_ARM_CALL                  = 0x1c
  R_ARM_JUMP24                = 0x1d
  R_ARM_THM_JUMP24            = 0x1e
  R_ARM_BASE_ABS              = 0x1f
  R_ARM_ALU_PCREL_7_0         = 0x20
  R_ARM_ALU_PCREL_15_8        = 0x21
  R_ARM_ALU_PCREL_23_15       = 0x22
  R_ARM_LDR_SBREL_11_0_NC     = 0x23
  R_ARM_ALU_SBREL_19_12_NC    = 0x24
  R_ARM_ALU_SBREL_27_20_CK    = 0x25
  R_ARM_TARGET1               = 0x26
  R_ARM_SBREL31               = 0x27
  R_ARM_V4BX                  = 0x28
  R_ARM_TARGET2               = 0x29
  R_ARM_PREL31                = 0x2a
  R_ARM_MOVW_ABS_NC           = 0x2b
  R_ARM_MOVT_ABS              = 0x2c
  R_ARM_MOVW_PREL_NC          = 0x2d
  R_ARM_MOVT_PREL             = 0x2e
  R_ARM_THM_MOVW_ABS_NC       = 0x2f
  R_ARM_THM_MOVT_ABS          = 0x30
  R_ARM_THM_MOVW_PREL_NC      = 0x31
  R_ARM_THM_MOVT_PREL         = 0x32
  R_ARM_THM_JUMP19            = 0x33
  R_ARM_THM_JUMP6             = 0x34
  R_ARM_THM_ALU_PREL_11_0     = 0x35
  R_ARM_THM_PC12              = 0x36
  R_ARM_ABS32_NOI             = 0x37
  R_ARM_REL32_NOI             = 0x38
  R_ARM_ALU_PC_G0_NC          = 0x39
  R_ARM_ALU_PC_G0             = 0x3a
  R_ARM_ALU_PC_G1_NC          = 0x3b
  R_ARM_ALU_PC_G1             = 0x3c
  R_ARM_ALU_PC_G2             = 0x3d
  R_ARM_LDR_PC_G1             = 0x3e
  R_ARM_LDR_PC_G2             = 0x3f
  R_ARM_LDRS_PC_G0            = 0x40
  R_ARM_LDRS_PC_G1            = 0x41
  R_ARM_LDRS_PC_G2            = 0x42
  R_ARM_LDC_PC_G0             = 0x43
  R_ARM_LDC_PC_G1             = 0x44
  R_ARM_LDC_PC_G2             = 0x45
  R_ARM_ALU_SB_G0_NC          = 0x46
  R_ARM_ALU_SB_G0             = 0x47
  R_ARM_ALU_SB_G1_NC          = 0x48
  R_ARM_ALU_SB_G1             = 0x49
  R_ARM_ALU_SB_G2             = 0x4a
  R_ARM_LDR_SB_G0             = 0x4b
  R_ARM_LDR_SB_G1             = 0x4c
  R_ARM_LDR_SB_G2             = 0x4d
  R_ARM_LDRS_SB_G0            = 0x4e
  R_ARM_LDRS_SB_G1            = 0x4f
  R_ARM_LDRS_SB_G2            = 0x50
  R_ARM_LDC_SB_G0             = 0x51
  R_ARM_LDC_SB_G1             = 0x52
  R_ARM_LDC_SB_G2             = 0x53
  R_ARM_MOVW_BREL_NC          = 0x54
  R_ARM_MOVT_BREL             = 0x55
  R_ARM_MOVW_BREL             = 0x56
  R_ARM_THM_MOVW_BREL_NC      = 0x57
  R_ARM_THM_MOVT_BREL         = 0x58
  R_ARM_THM_MOVW_BREL         = 0x59
  R_ARM_TLS_GOTDESC           = 0x5a
  R_ARM_TLS_CALL              = 0x5b
  R_ARM_TLS_DESCSEQ           = 0x5c
  R_ARM_THM_TLS_CALL          = 0x5d
  R_ARM_PLT32_ABS             = 0x5e
  R_ARM_GOT_ABS               = 0x5f
  R_ARM_GOT_PREL              = 0x60
  R_ARM_GOT_BREL12            = 0x61
  R_ARM_GOTOFF12              = 0x62
  R_ARM_GOTRELAX              = 0x63
  R_ARM_GNU_VTENTRY           = 0x64
  R_ARM_GNU_VTINHERIT         = 0x65
  R_ARM_THM_JUMP11            = 0x66
  R_ARM_THM_JUMP8             = 0x67
  R_ARM_TLS_GD32              = 0x68
  R_ARM_TLS_LDM32             = 0x69
  R_ARM_TLS_LDO32             = 0x6a
  R_ARM_TLS_IE32              = 0x6b
  R_ARM_TLS_LE32              = 0x6c
  R_ARM_TLS_LDO12             = 0x6d
  R_ARM_TLS_LE12              = 0x6e
  R_ARM_TLS_IE12GP            = 0x6f
  R_ARM_PRIVATE_0             = 0x70
  R_ARM_PRIVATE_1             = 0x71
  R_ARM_PRIVATE_2             = 0x72
  R_ARM_PRIVATE_3             = 0x73
  R_ARM_PRIVATE_4             = 0x74
  R_ARM_PRIVATE_5             = 0x75
  R_ARM_PRIVATE_6             = 0x76
  R_ARM_PRIVATE_7             = 0x77
  R_ARM_PRIVATE_8             = 0x78
  R_ARM_PRIVATE_9             = 0x79
  R_ARM_PRIVATE_10            = 0x7a
  R_ARM_PRIVATE_11            = 0x7b
  R_ARM_PRIVATE_12            = 0x7c
  R_ARM_PRIVATE_13            = 0x7d
  R_ARM_PRIVATE_14            = 0x7e
  R_ARM_PRIVATE_15            = 0x7f
  R_ARM_ME_TOO                = 0x80
  R_ARM_THM_TLS_DESCSEQ16     = 0x81
  R_ARM_THM_TLS_DESCSEQ32     = 0x82
  R_ARM_IRELATIVE             = 0xa0

class Relocs_Elf_Mips(Enum):
  R_MIPS_NONE              =  0
  R_MIPS_16                =  1
  R_MIPS_32                =  2
  R_MIPS_REL32             =  3
  R_MIPS_26                =  4
  R_MIPS_HI16              =  5
  R_MIPS_LO16              =  6
  R_MIPS_GPREL16           =  7
  R_MIPS_LITERAL           =  8
  R_MIPS_GOT16             =  9
  R_MIPS_PC16              = 10
  R_MIPS_CALL16            = 11
  R_MIPS_GPREL32           = 12
  R_MIPS_SHIFT5            = 16
  R_MIPS_SHIFT6            = 17
  R_MIPS_64                = 18
  R_MIPS_GOT_DISP          = 19
  R_MIPS_GOT_PAGE          = 20
  R_MIPS_GOT_OFST          = 21
  R_MIPS_GOT_HI16          = 22
  R_MIPS_GOT_LO16          = 23
  R_MIPS_SUB               = 24
  R_MIPS_INSERT_A          = 25
  R_MIPS_INSERT_B          = 26
  R_MIPS_DELETE            = 27
  R_MIPS_HIGHER            = 28
  R_MIPS_HIGHEST           = 29
  R_MIPS_CALL_HI16         = 30
  R_MIPS_CALL_LO16         = 31
  R_MIPS_SCN_DISP          = 32
  R_MIPS_REL16             = 33
  R_MIPS_ADD_IMMEDIATE     = 34
  R_MIPS_PJUMP             = 35
  R_MIPS_RELGOT            = 36
  R_MIPS_JALR              = 37
  R_MIPS_TLS_DTPMOD32      = 38
  R_MIPS_TLS_DTPREL32      = 39
  R_MIPS_TLS_DTPMOD64      = 40
  R_MIPS_TLS_DTPREL64      = 41
  R_MIPS_TLS_GD            = 42
  R_MIPS_TLS_LDM           = 43
  R_MIPS_TLS_DTPREL_HI16   = 44
  R_MIPS_TLS_DTPREL_LO16   = 45
  R_MIPS_TLS_GOTTPREL      = 46
  R_MIPS_TLS_TPREL32       = 47
  R_MIPS_TLS_TPREL64       = 48
  R_MIPS_TLS_TPREL_HI16    = 49
  R_MIPS_TLS_TPREL_LO16    = 50
  R_MIPS_GLOB_DAT          = 51
  R_MIPS_COPY              = 126
  R_MIPS_JUMP_SLOT         = 127
  R_MIPS_NUM               = 218

class Relocs_Elf_Hexagon(Enum):
  R_HEX_NONE              =  0
  R_HEX_B22_PCREL         =  1
  R_HEX_B15_PCREL         =  2
  R_HEX_B7_PCREL          =  3
  R_HEX_LO16              =  4
  R_HEX_HI16              =  5
  R_HEX_32                =  6
  R_HEX_16                =  7
  R_HEX_8                 =  8
  R_HEX_GPREL16_0         =  9
  R_HEX_GPREL16_1         =  10
  R_HEX_GPREL16_2         =  11
  R_HEX_GPREL16_3         =  12
  R_HEX_HL16              =  13
  R_HEX_B13_PCREL         =  14
  R_HEX_B9_PCREL          =  15
  R_HEX_B32_PCREL_X       =  16
  R_HEX_32_6_X            =  17
  R_HEX_B22_PCREL_X       =  18
  R_HEX_B15_PCREL_X       =  19
  R_HEX_B13_PCREL_X       =  20
  R_HEX_B9_PCREL_X        =  21
  R_HEX_B7_PCREL_X        =  22
  R_HEX_16_X              =  23
  R_HEX_12_X              =  24
  R_HEX_11_X              =  25
  R_HEX_10_X              =  26
  R_HEX_9_X               =  27
  R_HEX_8_X               =  28
  R_HEX_7_X               =  29
  R_HEX_6_X               =  30
  R_HEX_32_PCREL          =  31
  R_HEX_COPY              =  32
  R_HEX_GLOB_DAT          =  33
  R_HEX_JMP_SLOT          =  34
  R_HEX_RELATIVE          =  35
  R_HEX_PLT_B22_PCREL     =  36
  R_HEX_GOTREL_LO16       =  37
  R_HEX_GOTREL_HI16       =  38
  R_HEX_GOTREL_32         =  39
  R_HEX_GOT_LO16          =  40
  R_HEX_GOT_HI16          =  41
  R_HEX_GOT_32            =  42
  R_HEX_GOT_16            =  43
  R_HEX_DTPMOD_32         =  44
  R_HEX_DTPREL_LO16       =  45
  R_HEX_DTPREL_HI16       =  46
  R_HEX_DTPREL_32         =  47
  R_HEX_DTPREL_16         =  48
  R_HEX_GD_PLT_B22_PCREL  =  49
  R_HEX_GD_GOT_LO16       =  50
  R_HEX_GD_GOT_HI16       =  51
  R_HEX_GD_GOT_32         =  52
  R_HEX_GD_GOT_16         =  53
  R_HEX_IE_LO16           =  54
  R_HEX_IE_HI16           =  55
  R_HEX_IE_32             =  56
  R_HEX_IE_GOT_LO16       =  57
  R_HEX_IE_GOT_HI16       =  58
  R_HEX_IE_GOT_32         =  59
  R_HEX_IE_GOT_16         =  60
  R_HEX_TPREL_LO16        =  61
  R_HEX_TPREL_HI16        =  62
  R_HEX_TPREL_32          =  63
  R_HEX_TPREL_16          =  64
  R_HEX_6_PCREL_X         =  65
  R_HEX_GOTREL_32_6_X     =  66
  R_HEX_GOTREL_16_X       =  67
  R_HEX_GOTREL_11_X       =  68
  R_HEX_GOT_32_6_X        =  69
  R_HEX_GOT_16_X          =  70
  R_HEX_GOT_11_X          =  71
  R_HEX_DTPREL_32_6_X     =  72
  R_HEX_DTPREL_16_X       =  73
  R_HEX_DTPREL_11_X       =  74
  R_HEX_GD_GOT_32_6_X     =  75
  R_HEX_GD_GOT_16_X       =  76
  R_HEX_GD_GOT_11_X       =  77
  R_HEX_IE_32_6_X         =  78
  R_HEX_IE_16_X           =  79
  R_HEX_IE_GOT_32_6_X     =  80
  R_HEX_IE_GOT_16_X       =  81
  R_HEX_IE_GOT_11_X       =  82
  R_HEX_TPREL_32_6_X      =  83
  R_HEX_TPREL_16_X        =  84
  R_HEX_TPREL_11_X        =  85

class Relocs_Elf_Lanai(Enum):
  R_LANAI_NONE = 0
  R_LANAI_21   = 1
  R_LANAI_21_F = 2
  R_LANAI_25   = 3
  R_LANAI_32   = 4
  R_LANAI_HI16 = 5
  R_LANAI_LO16 = 6

class Relocs_Coff_i386(Enum):
  IMAGE_REL_I386_ABSOLUTE = 0x0000
  IMAGE_REL_I386_DIR16    = 0x0001
  IMAGE_REL_I386_REL16    = 0x0002
  IMAGE_REL_I386_DIR32    = 0x0006
  IMAGE_REL_I386_DIR32NB  = 0x0007
  IMAGE_REL_I386_SEG12    = 0x0009
  IMAGE_REL_I386_SECTION  = 0x000A
  IMAGE_REL_I386_SECREL   = 0x000B
  IMAGE_REL_I386_TOKEN    = 0x000C
  IMAGE_REL_I386_SECREL7  = 0x000D
  IMAGE_REL_I386_REL32    = 0x0014

class Relocs_Coff_X86_64(Enum):
  IMAGE_REL_AMD64_ABSOLUTE  = 0x0000
  IMAGE_REL_AMD64_ADDR64    = 0x0001
  IMAGE_REL_AMD64_ADDR32    = 0x0002
  IMAGE_REL_AMD64_ADDR32NB  = 0x0003
  IMAGE_REL_AMD64_REL32     = 0x0004
  IMAGE_REL_AMD64_REL32_1   = 0x0005
  IMAGE_REL_AMD64_REL32_2   = 0x0006
  IMAGE_REL_AMD64_REL32_3   = 0x0007
  IMAGE_REL_AMD64_REL32_4   = 0x0008
  IMAGE_REL_AMD64_REL32_5   = 0x0009
  IMAGE_REL_AMD64_SECTION   = 0x000A
  IMAGE_REL_AMD64_SECREL    = 0x000B
  IMAGE_REL_AMD64_SECREL7   = 0x000C
  IMAGE_REL_AMD64_TOKEN     = 0x000D
  IMAGE_REL_AMD64_SREL32    = 0x000E
  IMAGE_REL_AMD64_PAIR      = 0x000F
  IMAGE_REL_AMD64_SSPAN32   = 0x0010

class Relocs_Coff_ARM(Enum):
  IMAGE_REL_ARM_ABSOLUTE  = 0x0000
  IMAGE_REL_ARM_ADDR32    = 0x0001
  IMAGE_REL_ARM_ADDR32NB  = 0x0002
  IMAGE_REL_ARM_BRANCH24  = 0x0003
  IMAGE_REL_ARM_BRANCH11  = 0x0004
  IMAGE_REL_ARM_TOKEN     = 0x0005
  IMAGE_REL_ARM_BLX24     = 0x0008
  IMAGE_REL_ARM_BLX11     = 0x0009
  IMAGE_REL_ARM_SECTION   = 0x000E
  IMAGE_REL_ARM_SECREL    = 0x000F
  IMAGE_REL_ARM_MOV32A    = 0x0010
  IMAGE_REL_ARM_MOV32T    = 0x0011
  IMAGE_REL_ARM_BRANCH20T = 0x0012
  IMAGE_REL_ARM_BRANCH24T = 0x0014
  IMAGE_REL_ARM_BLX23T    = 0x0015


class Relocs_Macho_i386(Enum):
  RIT_Vanilla                     = 0
  RIT_Pair                        = 1
  RIT_Difference                  = 2
  RIT_Generic_PreboundLazyPointer = 3
  RIT_Generic_LocalDifference     = 4
  RIT_Generic_TLV                 = 5

class Relocs_Macho_X86_64(Enum):
  RIT_X86_64_Unsigned   = 0
  RIT_X86_64_Signed     = 1
  RIT_X86_64_Branch     = 2
  RIT_X86_64_GOTLoad    = 3
  RIT_X86_64_GOT        = 4
  RIT_X86_64_Subtractor = 5
  RIT_X86_64_Signed1    = 6
  RIT_X86_64_Signed2    = 7
  RIT_X86_64_Signed4    = 8
  RIT_X86_64_TLV        = 9

class Relocs_Macho_ARM(Enum):
  RIT_Vanilla                     = 0
  RIT_Pair                        = 1
  RIT_Difference                  = 2
  RIT_ARM_LocalDifference         = 3
  RIT_ARM_PreboundLazyPointer     = 4
  RIT_ARM_Branch24Bit             = 5
  RIT_ARM_ThumbBranch22Bit        = 6
  RIT_ARM_ThumbBranch32Bit        = 7
  RIT_ARM_Half                    = 8
  RIT_ARM_HalfDifference          = 9

class Relocs_Macho_PPC(Enum):
  PPC_RELOC_VANILLA        = 0
  PPC_RELOC_PAIR           = 1
  PPC_RELOC_BR14           = 2
  PPC_RELOC_BR24           = 3
  PPC_RELOC_HI16           = 4
  PPC_RELOC_LO16           = 5
  PPC_RELOC_HA16           = 6
  PPC_RELOC_LO14           = 7
  PPC_RELOC_SECTDIFF       = 8
  PPC_RELOC_PB_LA_PTR      = 9
  PPC_RELOC_HI16_SECTDIFF  = 10
  PPC_RELOC_LO16_SECTDIFF  = 11
  PPC_RELOC_HA16_SECTDIFF  = 12
  PPC_RELOC_JBSR           = 13
  PPC_RELOC_LO14_SECTDIFF  = 14
  PPC_RELOC_LOCAL_SECTDIFF = 15


craftElf("relocs.obj.elf-x86_64",   "x86_64-pc-linux-gnu",         Relocs_Elf_X86_64.entries(), "leaq sym@GOTTPOFF(%rip), %rax")
craftElf("relocs.obj.elf-i386",     "i386-pc-linux-gnu",           Relocs_Elf_i386.entries(),   "mov sym@GOTOFF(%ebx), %eax")
#craftElf("relocs-elf-ppc32",   "powerpc-unknown-linux-gnu",   Relocs_Elf_PPC32.entries(), ...)
craftElf("relocs.obj.elf-ppc64",   "powerpc64-unknown-linux-gnu", Relocs_Elf_PPC64.entries(),
         ("@t = thread_local global i32 0, align 4", "define i32* @f{0}() nounwind {{ ret i32* @t }}", 2))
craftElf("relocs.obj.elf-aarch64",  "aarch64",                     Relocs_Elf_AArch64.entries(), "movz x0, #:abs_g0:sym")
craftElf("relocs.obj.elf-aarch64-ilp32", "aarch64",
         Relocs_Elf_AArch64_ILP32.entries(), "movz x0, #:abs_g0:sym")
Relocs_Elf_AArch64_ILP32
craftElf("relocs.obj.elf-arm",      "arm-unknown-unknown",         Relocs_Elf_ARM.entries(), "b sym")
craftElf("relocs.obj.elf-mips",     "mips-unknown-linux",          Relocs_Elf_Mips.entries(), "lui $2, %hi(sym)")
craftElf("relocs.obj.elf-mips64el", "mips64el-unknown-linux",        Relocs_Elf_Mips.entries(), "lui $2, %hi(sym)")
#craftElf("relocs.obj.elf-hexagon",  "hexagon-unknown-unknown",     Relocs_Elf_Hexagon.entries(), ...)
#craftElf("relocs.obj.elf-lanai",   "lanai-unknown-unknown",   Relocs_Elf_Lanai.entries(), "mov hi(x), %r4")

craftCoff("relocs.obj.coff-i386",   "i386-pc-win32",   Relocs_Coff_i386.entries(),   "mov foo@imgrel(%ebx, %ecx, 4), %eax")
craftCoff("relocs.obj.coff-x86_64", "x86_64-pc-win32", Relocs_Coff_X86_64.entries(), "mov foo@imgrel(%ebx, %ecx, 4), %eax")
#craftCoff("relocs.obj.coff-arm",    "arm-pc-win32",    Relocs_Coff_ARM.entries(), "...")

craftMacho("relocs.obj.macho-i386",   "i386-apple-darwin9", Relocs_Macho_i386.entries(),
          ("asm", ".subsections_via_symbols; .text; a: ; b:", "call a", 1))
craftMacho("relocs.obj.macho-x86_64", "x86_64-apple-darwin9", Relocs_Macho_X86_64.entries(),
          ("asm", ".subsections_via_symbols; .text; a: ; b:", "call a", 1))
craftMacho("relocs.obj.macho-arm",    "armv7-apple-darwin10", Relocs_Macho_ARM.entries(), "bl sym")
#craftMacho("relocs.obj.macho-ppc",   "powerpc-apple-darwin10", Relocs_Macho_PPC.entries(), ...)

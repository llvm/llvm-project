## Disallow (de)compression for sections within a segment as they are
## effectively immutable.
# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: yaml2obj %s -o a
# RUN: not llvm-objcopy a /dev/null --compress-sections .text=zlib 2>&1 | FileCheck %s --implicit-check-not=error:

# CHECK: error: 'a': section '.text' within a segment cannot be (de)compressed

# RUN: not llvm-objcopy a /dev/null --compress-sections foo=none 2>&1 | FileCheck %s --check-prefix=CHECK2 --implicit-check-not=error:

# CHECK2: error: 'a': section 'foo' within a segment cannot be (de)compressed

## There is an error even if 'foo' is already compressed with zlib.
# RUN: not llvm-objcopy a /dev/null --compress-sections foo=zlib 2>&1 | FileCheck %s --check-prefix=CHECK3 --implicit-check-not=error:

# CHECK3: error: 'a': section 'foo' within a segment cannot be (de)compressed

--- !ELF
FileHeader:
  Class:      ELFCLASS64
  Data:       ELFDATA2LSB
  Type:       ET_EXEC
  Machine:    EM_X86_64
ProgramHeaders:
  - Type:     PT_LOAD
    FirstSec: .text
    LastSec:  foo
    Align:    0x1000
    Offset:   0x1000
Sections:
  - Name:     .text
    Type:     SHT_PROGBITS
    Offset:   0x1000
    Content:  C3
  - Name:     foo
    Type:     SHT_PROGBITS
    Flags:    [ SHF_COMPRESSED ]
    Content:  010000000000000040000000000000000100000000000000789cd36280002d3269002f800151

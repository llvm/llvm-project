// RUN: llvm-mc -triple x86_64-unknown-unknown -dwarf-version 5 -filetype=obj %s -o -| llvm-dwarfdump --debug-line - | FileCheck %s

        .file 1 "dir1/foo"   md5 "00112233445566778899aabbccddeeff"
        .file 2 "dir2" "bar" md5 "ffeeddccbbaa99887766554433221100"
        .loc 1 1 0
        nop
        .loc 2 1 0
        nop

# CHECK: debug_line[0x00000000]
# CHECK: version: 5
# CHECK: include_directories[ 0] = ''
# CHECK: include_directories[ 1] = 'dir1'
# CHECK: include_directories[ 2] = 'dir2'
# CHECK-NOT: include_directories
# CHECK: Dir MD5 Checksum File Name
# CHECK: file_names[ 1] 1 00112233445566778899aabbccddeeff foo
# CHECK: file_names[ 2] 2 ffeeddccbbaa99887766554433221100 bar

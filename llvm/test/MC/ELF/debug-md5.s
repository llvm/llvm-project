// RUN: llvm-mc -triple x86_64-unknown-unknown -dwarf-version 5 -fdebug-compilation-dir=/tmp -filetype=obj %s -o - | llvm-dwarfdump --debug-line --debug-line-str -v - | FileCheck %s

        .file 1 "dir1/foo"   md5 "00112233445566778899aabbccddeeff"
        .file 2 "dir2" "bar" md5 "ffeeddccbbaa99887766554433221100"
        .loc 1 1 0
        nop
        .loc 2 1 0
        nop

# CHECK: debug_line[0x00000000]
# CHECK: version: 5
# CHECK: include_directories[ 0] = .debug_line_str[0x[[DIR0:[0-9a-f]+]]] = "/tmp"
# CHECK: include_directories[ 1] = .debug_line_str[0x[[DIR1:[0-9a-f]+]]] = "dir1"
# CHECK: include_directories[ 2] = .debug_line_str[0x[[DIR2:[0-9a-f]+]]] = "dir2"
# CHECK-NOT: include_directories
# CHECK: file_names[ 0]:
# CHECK-NEXT: name: .debug_line_str[0x[[FILE0:[0-9a-f]+]]] = "{{.+}}"
# CHECK-NEXT: dir_index: 0
# CHECK: file_names[ 1]:
# CHECK-NEXT: name: .debug_line_str[0x[[FILE1:[0-9a-f]+]]] = "foo"
# CHECK-NEXT: dir_index: 1
# CHECK-NEXT: md5_checksum: 00112233445566778899aabbccddeeff
# CHECK: file_names[ 2]:
# CHECK-NEXT: name: .debug_line_str[0x[[FILE2:[0-9a-f]+]]] = "bar"
# CHECK-NEXT: dir_index: 2
# CHECK-NEXT: md5_checksum: ffeeddccbbaa99887766554433221100

# CHECK: .debug_line_str contents:
# CHECK-NEXT: 0x[[DIR0]]: "/tmp"
# CHECK-NEXT: 0x[[DIR1]]: "dir1"
# CHECK-NEXT: 0x[[DIR2]]: "dir2"
# CHECK-NEXT: 0x[[FILE0]]: "{{.+}}"
# CHECK-NEXT: 0x[[FILE1]]: "foo"
# CHECK-NEXT: 0x[[FILE2]]: "bar"

// RUN: llvm-mc -triple x86_64-unknown-unknown -dwarf-version 5 -filetype=obj %s -o -| llvm-dwarfdump --debug-line --debug-line-str -v - | FileCheck %s

// Test combinations of options to the .file directive.

        .file 1 "dir1/foo" md5 "ee87e05688663173cd6043a3a15bba6e" source "void foo() {}"
        .file 2 "dir2/bar" source "void bar() {}" md5 "816225a0c90ca8948b70eb58be4d522f"
        .loc 1 1 0
        nop
        .loc 2 1 0
        nop

# CHECK: debug_line[0x00000000]
# CHECK: version: 5
# CHECK: include_directories[ 0] = .debug_line_str[0x[[DIR0:[0-9a-f]+]]] = "{{.+}}"
# CHECK: include_directories[ 1] = .debug_line_str[0x[[DIR1:[0-9a-f]+]]] = "dir1"
# CHECK: include_directories[ 2] = .debug_line_str[0x[[DIR2:[0-9a-f]+]]] = "dir2"
# CHECK-NOT: include_directories
# CHECK: file_names[ 0]:
# CHECK-NEXT: name: .debug_line_str[0x[[FILE0:[0-9a-f]+]]] = "{{.+}}"
# CHECK-NEXT: dir_index: 0
# CHECK-NEXT: md5_checksum:
# CHECK-NEXT: source: .debug_line_str[0x[[FILE0SRC:[0-9a-f]+]]] = ""
# CHECK: file_names[ 1]:
# CHECK-NEXT: name: .debug_line_str[0x[[FILE1:[0-9a-f]+]]] = "foo"
# CHECK-NEXT: dir_index: 1
# CHECK-NEXT: md5_checksum: ee87e05688663173cd6043a3a15bba6e
# CHECK-NEXT: source: .debug_line_str[0x[[FILE1SRC:[0-9a-f]+]]] = "void foo() {}"
# CHECK: file_names[ 2]:
# CHECK-NEXT: name: .debug_line_str[0x[[FILE2:[0-9a-f]+]]] = "bar"
# CHECK-NEXT: dir_index: 2
# CHECK-NEXT: md5_checksum: 816225a0c90ca8948b70eb58be4d522f
# CHECK-NEXT: source: .debug_line_str[0x[[FILE2SRC:[0-9a-f]+]]] = "void bar() {}"

# CHECK: .debug_line_str contents:
# CHECK-NEXT: 0x[[DIR0]]: "{{.+}}"
# CHECK-NEXT: 0x[[DIR1]]: "dir1"
# CHECK-NEXT: 0x[[DIR2]]: "dir2"
# CHECK-NEXT: 0x[[FILE0]]: "{{.+}}"
# CHECK-NEXT: 0x[[FILE0SRC]]: ""
# CHECK-NEXT: 0x[[FILE1]]: "foo"
# CHECK-NEXT: 0x[[FILE1SRC]]: "void foo() {}"
# CHECK-NEXT: 0x[[FILE2]]: "bar"
# CHECK-NEXT: 0x[[FILE2SRC]]: "void bar() {}"

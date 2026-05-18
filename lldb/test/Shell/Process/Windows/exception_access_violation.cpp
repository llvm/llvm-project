// clang-format off

// REQUIRES: target-windows
// RUN: %build --compiler=clang-cl -o %t.exe -- %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -o "run" -- write | FileCheck --check-prefix=WRITE %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -o "run" -- read | FileCheck --check-prefix=READ %s

#include <string.h>

int access_violation_write(void* addr) {
    *(int*)addr = 42;
    return 0;
}


int access_violation_read(void* addr) {
    volatile int ret = *(int*)addr;
    return ret;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        return 1;
    }
    if (strcmp(argv[1], "write") == 0) {
        return access_violation_write((void*)42);
    }
    if (strcmp(argv[1], "read") == 0) {
        return access_violation_read((void*)42);
    }
    return 1;
}


// WRITE:     * thread #1, stop reason = Exception 0xc0000005 encountered at address {{.*}}: Access violation writing location 0x0000002a

// READ:      * thread #1, stop reason = Exception 0xc0000005 encountered at address {{.*}}: Access violation reading location 0x0000002a

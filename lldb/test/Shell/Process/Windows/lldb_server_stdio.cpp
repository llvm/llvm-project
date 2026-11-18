// clang-format off

// lldb-server's stdio is the null device, so nothing it prints ends up in
// lldb's output.

// REQUIRES: target-windows
// RUN: %build -o %t.exe -- %s
// RUN: env LLDB_USE_LLDB_SERVER=1 %lldb -b -o run -f %t.exe 2>&1 | \
// RUN:   FileCheck %s --implicit-check-not="Connection established" \
// RUN:     --implicit-check-not="lldb-server exiting"

// CHECK: Process {{[0-9]+}} exited with status = 0

int main() { return 0; }

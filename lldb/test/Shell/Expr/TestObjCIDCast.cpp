// RUN: %clangxx_host %s -g -o %t
//
// RUN: %lldb -f %t \
// RUN:  -o "settings set interpreter.stop-command-source-on-error false" \
// RUN:  -o "b main" -o run -o "expression --language objc -- *(id)0x1234" -o exit 2>&1 | FileCheck %s

int main() { return 0; }

// CHECK: (lldb) expression --language objc -- *(id)0x1234
// CHECK: error: Couldn't apply expression side effects : Couldn't dematerialize a result variable: couldn't read its memory

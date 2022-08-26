// Test CommandObject is cleaned up even after commands fail due to not taking any argument.
// RUN: %clang_host -g %s -o %t
// RUN: %lldb -f %t -o "settings set interpreter.stop-command-source-on-error false" -s \
// RUN:   %S/Inputs/cleanup.lldbinit
int main() { return 0; }

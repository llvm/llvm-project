// REQUIRES: target-windows
// RUN: %build --compiler=clang-cl -o %t.exe -- %s
// RUN: %lldb -f %t.exe -o "log enable windows process" -o r -o q | FileCheck %s

#include <Windows.h>
#include <string>

int main() {
  OutputDebugStringA("My string\nnext line\nBut this doesn't have trailing newline|");
  OutputDebugStringW(L"OutputDebugStringW with some emojis 🦎🐊🐢🐍\n");
  OutputDebugStringW(L"Another W1\n");
  OutputDebugStringW(L"Another W2\n");
  OutputDebugStringA("Some A1\n");
  OutputDebugStringA("Some A2\n");
  OutputDebugStringA("Some A3\n");

  std::wstring maxW((1 << 19) - 4, L'A');
  maxW.push_back(L'B');
  maxW.push_back(L'\n');
  OutputDebugStringW(maxW.c_str());

  std::string maxA((1 << 20) - 4, 'C');
  maxA.push_back(L'D');
  maxA.push_back(L'\n');
  OutputDebugStringA(maxA.c_str());

  // Give LLDB time to print out the previous debug strings.
  // The following ones should generate a log.
  // This makes sure we see the log after the previous line, not before.
  Sleep(100);

  std::wstring tooBigW((1 << 19) - 3, L'E');
  tooBigW.push_back(L'F');
  tooBigW.push_back(L'\n');
  OutputDebugStringW(tooBigW.c_str());

  Sleep(100);

  std::string tooBigA((1 << 20) - 3, 'G');
  tooBigA.push_back(L'H');
  tooBigA.push_back(L'\n');
  OutputDebugStringA(tooBigA.c_str());

  return 0;
}

// CHECK: My string
// CHECK-NEXT: next line
// CHECK-NEXT: But this doesn't have trailing newline|OutputDebugStringW with some emojis 🦎🐊🐢🐍
// CHECK-NEXT: Another W1
// CHECK-NEXT: Another W2
// CHECK-NEXT: Some A1
// CHECK-NEXT: Some A2
// CHECK-NEXT: Some A3
// CHECK-NEXT: {{A+B}}
// CHECK-NEXT: {{C+D}}
// CHECK-NEXT: Failed to read debug string at 0x{{.*}} (size & 0xffff=0, unicode=true): String is 1 MiB or larger
// CHECK-NEXT: Failed to read debug string at 0x{{.*}} (size & 0xffff=0, unicode=false): String is 1 MiB or larger

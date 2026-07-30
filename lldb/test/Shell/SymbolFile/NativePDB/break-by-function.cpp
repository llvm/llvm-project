// clang-format off
// REQUIRES: lld, x86

// Test that we can set simple breakpoints using PDB on any platform.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: %lldb -f %t.exe -s \
// RUN:     %p/Inputs/break-by-function.lldbinit | FileCheck %s

// Use different indentation style for each overload so that the starting
// line is in a different place.

int OvlGlobalFn(int X) {
  return X + 42;
}

int OvlGlobalFn(int X, int Y) { return X + Y + 42; }

int OvlGlobalFn(int X, int Y, int Z)
{
  return X + Y + Z + 42;
}

static int StaticFn(int X) {
  return X + 42;
}

int main(int argc, char **argv) {
  // Make sure they don't get optimized out.
  // Note the comments here, we want to make sure the line number reported
  // for the breakpoint is the first actual line of code.
  int Result = OvlGlobalFn(argc) + OvlGlobalFn(argc, argc)
    + OvlGlobalFn(argc, argc, argc) + StaticFn(argc);
  return Result;
}


// CHECK:      (lldb) target create "{{.*}}break-by-function.cpp.tmp.exe"
// CHECK:      Current executable set to '{{.*}}break-by-function.cpp.tmp.exe'
// CHECK:      (lldb) break set -n main
// CHECK:      Breakpoint 1: where = break-by-function.cpp.tmp.exe`main + {{[0-9]+}}
// CHECK:      (lldb) break set -n OvlGlobalFn
// CHECK:      Breakpoint 2: 3 locations.
// CHECK:      (lldb) break set -n StaticFn
// CHECK:      Breakpoint 3: where = break-by-function.cpp.tmp.exe`StaticFn + {{[0-9]+}}
// CHECK:      (lldb) break set -n DoesntExist
// CHECK:      Breakpoint 4: no locations (pending).
// CHECK:      (lldb) break list
// CHECK:      Current breakpoints:
// CHECK:      1: name = 'main', locations = 1
// CHECK:        1.1: where = break-by-function.cpp.tmp.exe`main + {{[0-9]+}}
// CHECK:      2: name = 'OvlGlobalFn', locations = 3
// CHECK:        2.1: where = break-by-function.cpp.tmp.exe`int OvlGlobalFn(int) + {{[0-9]+}}
// CHECK:        2.2: where = break-by-function.cpp.tmp.exe`int OvlGlobalFn(int, int)
// CHECK:        2.3: where = break-by-function.cpp.tmp.exe`int OvlGlobalFn(int, int, int) + {{[0-9]+}}
// CHECK:      3: name = 'StaticFn', locations = 1
// CHECK:        3.1: where = break-by-function.cpp.tmp.exe`StaticFn + {{[0-9]+}}
// CHECK:      4: name = 'DoesntExist', locations = 0 (pending)

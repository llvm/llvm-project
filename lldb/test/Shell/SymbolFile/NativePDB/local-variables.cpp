// clang-format off

// REQUIRES: target-windows
// RUN: %build -o %t.exe -- %s
// RUN: %lldb -f %t.exe -s \
// RUN:     %p/Inputs/local-variables.lldbinit 2>&1 | FileCheck %s

int Function(int Param1, char Param2) {
  unsigned Local1 = Param1 + 1;
  char Local2 = Param2 + 1;
  ++Local1;
  ++Local2;
  return Local1;
}

int main(int argc, char **argv) {
  int SomeLocal = argc * 2;
  return Function(SomeLocal, 'a');
}

// CHECK:      (lldb) target create "{{.*}}local-variables.cpp.tmp.exe"
// CHECK-DAG: Current executable set to '{{.*}}local-variables.cpp.tmp.exe'
// CHECK-DAG: (lldb) command source -s 0 '{{.*}}local-variables.lldbinit'
// CHECK-DAG: Executing commands in '{{.*}}local-variables.lldbinit'.
// CHECK-DAG: (lldb) break set -f local-variables.cpp -l 17
// CHECK-DAG: Breakpoint 1: where = local-variables.cpp.tmp.exe`main + {{.*}} at local-variables.cpp:{{.*}}, address = {{.*}}
// CHECK-DAG: (lldb) run a b c d e f g
// CHECK-DAG: Process {{.*}} launched: '{{.*}}local-variables.cpp.tmp.exe'
// CHECK-DAG: Process {{.*}} stopped
// CHECK-DAG: * thread #1, stop reason = breakpoint 1.1
// CHECK-DAG:     frame #0: {{.*}} local-variables.cpp.tmp.exe`main(argc=8, argv={{.*}}) at local-variables.cpp:{{.*}}
// CHECK-DAG:    14   }
// CHECK-DAG:    15
// CHECK-DAG:    16   int main(int argc, char **argv) {
// CHECK-DAG: -> 17     int SomeLocal = argc * 2;
// CHECK-DAG:    18     return Function(SomeLocal, 'a');
// CHECK-DAG:    19   }
// CHECK-DAG:    20

// CHECK:      (lldb) expression argc
// CHECK-DAG: (int) $0 = 8
// CHECK-DAG: (lldb) step
// CHECK-DAG: Process {{.*}} stopped
// CHECK-DAG: * thread #1, stop reason = step in
// CHECK-DAG:     frame #0: {{.*}} local-variables.cpp.tmp.exe`main(argc=8, argv={{.*}}) at local-variables.cpp:{{.*}}
// CHECK-DAG:    15
// CHECK-DAG:    16 int main(int argc, char **argv) {
// CHECK-DAG:    17     int SomeLocal = argc * 2;
// CHECK-DAG: -> 18     return Function(SomeLocal, 'a');
// CHECK-DAG:    19 }
// CHECK-DAG:    20

// CHECK:      (lldb) expression SomeLocal
// CHECK-DAG: (int) $1 = 16
// CHECK-DAG: (lldb) step
// CHECK-DAG: Process {{.*}} stopped
// CHECK-DAG: * thread #1, stop reason = step in
// CHECK-DAG:     frame #0: {{.*}} local-variables.cpp.tmp.exe`int Function(Param1=16, Param2='a') at local-variables.cpp:{{.*}}
// CHECK-DAG:    6
// CHECK-DAG:    7
// CHECK-DAG:    8 int Function(int Param1, char Param2) {
// CHECK-DAG: -> 9      unsigned Local1 = Param1 + 1;
// CHECK-DAG:    10     char Local2 = Param2 + 1;
// CHECK-DAG:    11     ++Local1;
// CHECK-DAG:    12     ++Local2;

// CHECK:      (lldb) expression Param1
// CHECK-DAG: (int) $2 = 16
// CHECK-DAG: (lldb) expression Param2
// CHECK-DAG: (char) $3 = 'a'
// CHECK-DAG: (lldb) step
// CHECK-DAG: Process {{.*}} stopped
// CHECK-DAG: * thread #1, stop reason = step in
// CHECK-DAG:     frame #0: {{.*}} local-variables.cpp.tmp.exe`int Function(Param1=16, Param2='a') at local-variables.cpp:{{.*}}
// CHECK-DAG:    7
// CHECK-DAG:    8    int Function(int Param1, char Param2) {
// CHECK-DAG:    9      unsigned Local1 = Param1 + 1;
// CHECK-DAG: -> 10     char Local2 = Param2 + 1;
// CHECK-DAG:    11     ++Local1;
// CHECK-DAG:    12     ++Local2;
// CHECK-DAG:    13     return Local1;

// CHECK:      (lldb) expression Param1
// CHECK-DAG: (int) $4 = 16
// CHECK-DAG: (lldb) expression Param2
// CHECK-DAG: (char) $5 = 'a'
// CHECK-DAG: (lldb) expression Local1
// CHECK-DAG: (unsigned int) $6 = 17
// CHECK-DAG: (lldb) step
// CHECK-DAG: Process {{.*}} stopped
// CHECK-DAG: * thread #1, stop reason = step in
// CHECK-DAG:     frame #0: {{.*}} local-variables.cpp.tmp.exe`int Function(Param1=16, Param2='a') at local-variables.cpp:{{.*}}
// CHECK-DAG:    8    int Function(int Param1, char Param2) {
// CHECK-DAG:    9      unsigned Local1 = Param1 + 1;
// CHECK-DAG:    10     char Local2 = Param2 + 1;
// CHECK-DAG: -> 11     ++Local1;
// CHECK-DAG:    12     ++Local2;
// CHECK-DAG:    13     return Local1;
// CHECK-DAG:    14   }

// CHECK:      (lldb) expression Param1
// CHECK-DAG: (int) $7 = 16
// CHECK-DAG: (lldb) expression Param2
// CHECK-DAG: (char) $8 = 'a'
// CHECK-DAG: (lldb) expression Local1
// CHECK-DAG: (unsigned int) $9 = 17
// CHECK-DAG: (lldb) expression Local2
// CHECK-DAG: (char) $10 = 'b'
// CHECK-DAG: (lldb) step
// CHECK-DAG: Process {{.*}} stopped
// CHECK-DAG: * thread #1, stop reason = step in
// CHECK-DAG:     frame #0: {{.*}} local-variables.cpp.tmp.exe`int Function(Param1=16, Param2='a') at local-variables.cpp:{{.*}}
// CHECK-DAG:    9      unsigned Local1 = Param1 + 1;
// CHECK-DAG:    10     char Local2 = Param2 + 1;
// CHECK-DAG:    11     ++Local1;
// CHECK-DAG: -> 12     ++Local2;
// CHECK-DAG:    13     return Local1;
// CHECK-DAG:    14   }
// CHECK-DAG:    15

// CHECK:      (lldb) expression Param1
// CHECK-DAG: (int) $11 = 16
// CHECK-DAG: (lldb) expression Param2
// CHECK-DAG: (char) $12 = 'a'
// CHECK-DAG: (lldb) expression Local1
// CHECK-DAG: (unsigned int) $13 = 18
// CHECK-DAG: (lldb) expression Local2
// CHECK-DAG: (char) $14 = 'b'
// CHECK-DAG: (lldb) step
// CHECK-DAG: Process {{.*}} stopped
// CHECK-DAG: * thread #1, stop reason = step in
// CHECK-DAG:     frame #0: {{.*}} local-variables.cpp.tmp.exe`int Function(Param1=16, Param2='a') at local-variables.cpp:{{.*}}
// CHECK-DAG:    10      char Local2 = Param2 + 1;
// CHECK-DAG:    11     ++Local1;
// CHECK-DAG:    12     ++Local2;
// CHECK-DAG: -> 13     return Local1;
// CHECK-DAG:    14   }
// CHECK-DAG:    15
// CHECK-DAG:    16   int main(int argc, char **argv) {

// CHECK:      (lldb) expression Param1
// CHECK-DAG: (int) $15 = 16
// CHECK-DAG: (lldb) expression Param2
// CHECK-DAG: (char) $16 = 'a'
// CHECK-DAG: (lldb) expression Local1
// CHECK-DAG: (unsigned int) $17 = 18
// CHECK-DAG: (lldb) expression Local2
// CHECK-DAG: (char) $18 = 'c'
// CHECK-DAG: (lldb) continue
// CHECK-DAG: Process {{.*}} resuming
// CHECK-DAG: Process {{.*}} exited with status = 18 (0x00000012)

// CHECK:      (lldb) target modules dump ast
// CHECK-DAG: Dumping clang ast for {{.*}} modules.
// CHECK-DAG: TranslationUnitDecl
// CHECK-DAG: |-FunctionDecl {{.*}} main 'int (int, char **)'
// CHECK-DAG: | |-ParmVarDecl {{.*}} argc 'int'
// CHECK-DAG: | `-ParmVarDecl {{.*}} argv 'char **'
// CHECK-DAG: |-FunctionDecl {{.*}} __scrt_common_main_seh 'int ()' static 
// CHECK-DAG: |-FunctionDecl {{.*}} invoke_main 'int ()' inline
// CHECK: `-FunctionDecl {{.*}} Function 'int (int, char)'
// CHECK-DAG:   |-ParmVarDecl {{.*}} Param1 'int'
// CHECK-DAG:   `-ParmVarDecl {{.*}} Param2 'char'

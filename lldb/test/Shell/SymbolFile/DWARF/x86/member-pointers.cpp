// Itanium ABI:
// RUN: %clang --target=x86_64-pc-linux -gdwarf -c -o %t_linux.o %s
// RUN: %lldb -f %t_linux.o -b -o "target variable mp1 mp2 mp3 mp4 mp5 mp6 mp7 mp8 mp9" | FileCheck %s
//
// CHECK: (char SI2::*) mp9 = 0x0000000000000000

// Microsoft ABI:
// RUN: %clang_cl --target=x86_64-windows-msvc -c -gdwarf %s -o %t_win.obj
// RUN: lld-link /out:%t_win.exe %t_win.obj /nodefaultlib /entry:main /debug
// RUN: %lldb -f %t_win.exe -b -o "target variable mp1 mp2 mp3 mp4 mp5 mp6 mp7 mp8 mp9" | FileCheck --check-prefix=CHECK-MSVC %s
//
// DWARF has no representation of MSInheritanceAttr, so we cannot determine the size
// of member-pointers yet. For the moment, make sure we don't crash on such variables.
// CHECK-MSVC: error: Unable to determine byte size.

class SI {
  double si;
};
struct SI2 {
  char si2;
};
class MI : SI, SI2 {
  int mi;
};
class MI2 : MI {
  int mi2;
};
class VI : virtual MI {
  int vi;
};
class VI2 : virtual SI, virtual SI2 {
  int vi;
};
class /* __unspecified_inheritance*/ UI;

double SI::*mp1 = nullptr;
int MI::*mp2 = nullptr;
int MI2::*mp3 = nullptr;
int VI::*mp4 = nullptr;
int VI2::*mp5 = nullptr;
int UI::*mp6 = nullptr;
int MI::*mp7 = nullptr;
int VI2::*mp8 = nullptr;
char SI2::*mp9 = &SI2::si2;

int main() { return 0; }

// When linking with link.exe, -gdwarf still produces PDB instead.
// UNSUPPORTED: system-windows

// RUN: %clangxx_host -gdwarf -std=c++11 -o %t %s
// RUN: %lldb %t \
// RUN:   -o run \
// RUN:   -o "frame var --show-types f" \
// RUN:   -o "frame var vla0" \
// RUN:   -o "frame var fla0" \
// RUN:   -o "frame var fla1" \
// RUN:   -o "frame var vla01" \
// RUN:   -o "frame var vla10" \
// RUN:   -o "frame var vlaN" \
// RUN:   -o "frame var vlaNM" \
// RUN:   -o exit | FileCheck %s

struct Foo {
  static constexpr int n = 1;
  int m_vlaN[n];

  int m_vla0[0];
};

int main() {
  Foo f;
  f.m_vlaN[0] = 60;

  // CHECK:      (lldb) frame var --show-types f
  // CHECK-NEXT: (Foo) f = {
  // CHECK-NEXT:   (int[1]) m_vlaN = {
  // CHECK-NEXT:     (int) [0] = 60
  // CHECK-NEXT:   }
  // CHECK-NEXT:   (int[0]) m_vla0 = {}
  // CHECK-NEXT: }

  int vla0[0] = {};

  // CHECK:      (lldb) frame var vla0
  // CHECK-NEXT: (int[0]) vla0 = {}

  int fla0[] = {};

  // CHECK:      (lldb) frame var fla0
  // CHECK-NEXT: (int[0]) fla0 = {}

  int fla1[] = {42};

  // CHECK:      (lldb) frame var fla1
  // CHECK-NEXT: (int[1]) fla1 = ([0] = 42)

  int vla01[0][1];

  // CHECK:      (lldb) frame var vla01
  // CHECK-NEXT: (int[0][1]) vla01 = {}

  int vla10[1][0];

  // CHECK:      (lldb) frame var vla10
  // CHECK-NEXT: (int[1][0]) vla10 = ([0] = int[0]

  int n = 3;
  int vlaN[n];
  for (int i = 0; i < n; ++i)
    vlaN[i] = -i;

  // CHECK:      (lldb) frame var vlaN
  // CHECK-NEXT: (int[]) vlaN = ([0] = 0, [1] = -1, [2] = -2)

  int m = 2;
  int vlaNM[n][m];
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      vlaNM[i][j] = i + j;

  // FIXME: multi-dimensional VLAs aren't well supported
  // CHECK:      (lldb) frame var vlaNM
  // CHECK-NEXT: (int[][]) vlaNM = {
  // CHECK-NEXT:   [0] = ([0] = 0, [1] = 1, [2] = 1)
  // CHECK-NEXT:   [1] = ([0] = 1, [1] = 1, [2] = 2)
  // CHECK-NEXT: }

  __builtin_debugtrap();
}

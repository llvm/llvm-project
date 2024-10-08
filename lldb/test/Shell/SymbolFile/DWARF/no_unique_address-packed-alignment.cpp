// RUN: %clangxx_host -gdwarf -o %t %s
// RUN: %lldb %t \
// RUN:   -o "expr alignof(PackedNUA)" \
// RUN:   -o "expr sizeof(PackedNUA)" \
// RUN:   -o exit | FileCheck %s

struct Empty {};
struct __attribute((packed)) PackedNUA {
  [[no_unique_address]] Empty a, b, c, d;
  char x;
  int y;
};
static_assert(alignof(PackedNUA) == 1);
static_assert(sizeof(PackedNUA) == 5);

PackedNUA packed;

// CHECK:      (lldb) expr alignof(PackedNUA)
// CHECK-NEXT: ${{.*}} = 1
// CHECK:      (lldb) expr sizeof(PackedNUA)
// CHECK-NEXT: ${{.*}} = 5

int main() { PackedNUA d; }

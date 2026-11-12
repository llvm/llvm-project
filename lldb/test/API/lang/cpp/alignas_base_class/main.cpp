struct B1 {
  char f1;
};

struct alignas(8) B2 {
  char f2;
};

struct D : B1, B2 {};

D d3g;

struct alignas(8) EmptyClassAlign8 {
} t;

struct alignas(8) __attribute__((packed)) AlignedAndPackedBase {
} foo;

struct Derived : AlignedAndPackedBase {
} bar;
static_assert(alignof(Derived) == 8);

int main() {}

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

int main() {}

struct B1 {
  char f1;
};

struct alignas(8) B2 {
  char f2;
};

struct D : B1, B2 {};

D d3g;

int main() {}

// Rank > 0 array
typedef volatile int* RankNArray[10][100];
RankNArray ArrayVar;

typedef int __unaligned *UnalignedTypedef;
UnalignedTypedef UnVar;

typedef long* __restrict RestrictTypedef;
RestrictTypedef RestrictVar;

void Func1(const int* a, int const* b, const int ** const c, const int* const* d) {
  return;
}

void Func2(volatile int* a, int volatile* b) {
 return;
}

void Func3(int*& a, int& b, const int&c, int&& d) {
  return;
}

void Func4(int* __unaligned a, __unaligned int* b) {
  return;
}

void Func5(int a, int* __restrict b, int& __restrict c) {
  return;
}

void Func6(const volatile int* __restrict b) {
  return;
}

// LValue
typedef int& IntRef;
int x = 0;
IntRef IVar = x;

// RValue
typedef int&& IIRef;
IIRef IIVar = int(1);

int main() {
  return 0;
}

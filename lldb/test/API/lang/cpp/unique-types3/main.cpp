#include "a.h"

S<double> a1;
S<int> a2;
S<float> a3;

void f(S<int> &);

int main() { f(a2); }

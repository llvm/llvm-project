// Compiled on macOS using:
// clang++ -c -std=c++2a -gdwarf-4 -O0 -o accel-imported-declaration.macho-arm64.o

namespace A {
namespace B {
namespace C {
int a = -1;
} // namespace C
} // namespace B

namespace C = B::C;

using namespace B::C;
using B::C::a;
} // namespace A

int main() { return A::a; }

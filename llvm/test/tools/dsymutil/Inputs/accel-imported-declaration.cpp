// Compiled on macOS using:
// 1. clang++ -c -std=c++2a -gdwarf-4 -O0 -o accel-imported-declaration.macho-arm64.o
// 2. clang++ -Wl,-oso_prefix=$PWD accel-imported-declaration.macho-arm64.o -o accel-imported-declaration.macho-arm64
//
// In step 2 it's important to strip the absolute object file paths
//
// Verify that the OSO path isn't absolute using `nm -ap accel-imported-declaration.macho-arm64`

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

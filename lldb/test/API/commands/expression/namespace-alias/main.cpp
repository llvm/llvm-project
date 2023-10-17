namespace A {
inline namespace _A {
namespace B {
namespace C {
int a = -1;

int func() { return 0; }
} // namespace C
} // namespace B

namespace C = B::C;
namespace D = B::C;

} // namespace _A
} // namespace A

namespace E = A;
namespace F = E::C;
namespace G = F;

int main(int argc, char **argv) { return A::B::C::a; }

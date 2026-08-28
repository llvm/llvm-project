struct A {
    constexpr A(int x) : val(x) {}
    int val;
};
struct B : A {
    constexpr int f() const { return 42; }
    constexpr B() : A(f()) {}
};
constexpr int foo() {
    constexpr B b{};
    return b.val;
}
int main() {
    constexpr auto x = foo();
    return 0;
}

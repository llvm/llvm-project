// clang-format off
// REQUIRES: lld, x86

// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/function-types-builtins.lldbinit | FileCheck %s

// Test that we can display function signatures with simple builtin
// and pointer types.  We do this by using `target variable` in lldb
// with global variables of type ptr-to-function or reference-to-function.
// This technique in general allows us to exercise most of LLDB's type
// system without a running process.

// Define _fltused, since we're not linking against the MS C runtime, but use
// floats.
extern "C" int _fltused = 0;

template<typename T>
struct MakeResult {
  static T result() {
    return T{};
  }
};

template<typename T>
struct MakeResult<T&> {
  static T& result() {
    static T t;
    return t;
  }
};

template<typename T>
struct MakeResult<T&&> {
  static T&& result() {
    static T t;
    return static_cast<T&&>(t);
  }
};


void nullary() {}

template<typename Arg>
void unary(Arg) { }

template<typename Ret, int N>
Ret unaryret() { return MakeResult<Ret>::result(); }

template<typename A1, typename A2>
void binary(A1, A2) { }

int varargs(int, int, ...) { return 0; }

// Make sure to test every builtin type at least once for completeness.  We
// test these in the globals-fundamentals.cpp when they are the types of
// variables but it's possible to imagine a situation where things behave
// differently as function arguments or return values than they do with
// global variables.

// some interesting cases with argument types.
auto aa = &unary<bool>;
// CHECK: (void (*)(bool)) aa = {{.*}}
auto ab = &unary<char>;
// CHECK: (void (*)(char)) ab = {{.*}}
auto ac = &unary<signed char>;
// CHECK: (void (*)(signed char)) ac = {{.*}}
auto ad = &unary<unsigned char>;
// CHECK: (void (*)(unsigned char)) ad = {{.*}}
auto ae = &unary<char16_t>;
// CHECK: (void (*)(char16_t)) ae = {{.*}}
auto af = &unary<char32_t>;
// CHECK: (void (*)(char32_t)) af = {{.*}}
auto ag = &unary<wchar_t>;
// CHECK: (void (*)(wchar_t)) ag = {{.*}}
auto ah = &unary<short>;
// CHECK: (void (*)(short)) ah = {{.*}}
auto ai = &unary<unsigned short>;
// CHECK: (void (*)(unsigned short)) ai = {{.*}}
auto aj = &unary<int>;
// CHECK: (void (*)(int)) aj = {{.*}}
auto ak = &unary<unsigned int>;
// CHECK: (void (*)(unsigned int)) ak = {{.*}}
auto al = &unary<long>;
// CHECK: (void (*)(long)) al = {{.*}}
auto am = &unary<unsigned long>;
// CHECK: (void (*)(unsigned long)) am = {{.*}}
auto an = &unary<long long>;
// CHECK: (void (*)(long long)) an = {{.*}}
auto ao = &unary<unsigned long long>;
// CHECK: (void (*)(unsigned long long)) ao = {{.*}}
auto aq = &unary<float>;
// CHECK: (void (*)(float)) aq = {{.*}}
auto ar = &unary<double>;
// CHECK: (void (*)(double)) ar = {{.*}}

auto as = &unary<int*>;
// CHECK: (void (*)(int *)) as = {{.*}}
auto at = &unary<int**>;
// CHECK: (void (*)(int **)) at = {{.*}}
auto au = &unary<int&>;
// CHECK: (void (*)(int &)) au = {{.*}}
auto av = &unary<int&&>;
// CHECK: (void (*)(int &&)) av = {{.*}}
auto aw = &unary<const int*>;
// CHECK: (void (*)(const int *)) aw = {{.*}}
auto ax = &unary<volatile int*>;
// CHECK: (void (*)(volatile int *)) ax = {{.*}}
auto ay = &unary<const volatile int*>;
// CHECK: (void (*)(const volatile int *)) ay = {{.*}}
auto az = &unary<void*&>;
// CHECK: (void (*)(void *&)) az = {{.*}}
auto aaa = &unary<int(&)[5]>;
// CHECK: (void (*)(int (&)[5])) aaa = {{.*}}
auto aab = &unary<int(*)[5]>;
// CHECK: (void (*)(int (*)[5])) aab = {{.*}}
auto aac = &unary<int(&&)[5]>;
// CHECK: (void (*)(int (&&)[5])) aac = {{.*}}
auto aad = &unary<int(*const)[5]>;
// CHECK: (void (*)(int (*const)[5])) aad = {{.*}}


// same test cases with return values, note we can't overload on return type
// so we need to use a different instantiation each time.
auto ra = &unaryret<bool, 0>;
// CHECK: (bool (*)()) ra = {{.*}}
auto rb = &unaryret<char, 1>;
// CHECK: (char (*)()) rb = {{.*}}
auto rc = &unaryret<signed char, 2>;
// CHECK: (signed char (*)()) rc = {{.*}}
auto rd = &unaryret<unsigned char, 3>;
// CHECK: (unsigned char (*)()) rd = {{.*}}
auto re = &unaryret<char16_t, 4>;
// CHECK: (char16_t (*)()) re = {{.*}}
auto rf = &unaryret<char32_t, 5>;
// CHECK: (char32_t (*)()) rf = {{.*}}
auto rg = &unaryret<wchar_t, 6>;
// CHECK: (wchar_t (*)()) rg = {{.*}}
auto rh = &unaryret<short, 7>;
// CHECK: (short (*)()) rh = {{.*}}
auto ri = &unaryret<unsigned short, 8>;
// CHECK: (unsigned short (*)()) ri = {{.*}}
auto rj = &unaryret<int, 9>;
// CHECK: (int (*)()) rj = {{.*}}
auto rk = &unaryret<unsigned int, 10>;
// CHECK: (unsigned int (*)()) rk = {{.*}}
auto rl = &unaryret<long, 11>;
// CHECK: (long (*)()) rl = {{.*}}
auto rm = &unaryret<unsigned long, 12>;
// CHECK: (unsigned long (*)()) rm = {{.*}}
auto rn = &unaryret<long long, 13>;
// CHECK: (long long (*)()) rn = {{.*}}
auto ro = &unaryret<unsigned long long, 14>;
// CHECK: (unsigned long long (*)()) ro = {{.*}}
auto rq = &unaryret<float, 15>;
// CHECK: (float (*)()) rq = {{.*}}
auto rr = &unaryret<double, 16>;
// CHECK: (double (*)()) rr = {{.*}}

auto rs = &unaryret<int*, 17>;
// CHECK: (int *(*)()) rs = {{.*}}
auto rt = &unaryret<int**, 18>;
// CHECK: (int **(*)()) rt = {{.*}}
auto ru = &unaryret<int&, 19>;
// CHECK: (int &(*)()) ru = {{.*}}
auto rv = &unaryret<int&&, 20>;
// CHECK: (int &&(*)()) rv = {{.*}}
auto rw = &unaryret<const int*, 21>;
// CHECK: (const int *(*)()) rw = {{.*}}
auto rx = &unaryret<volatile int*, 22>;
// CHECK: (volatile int *(*)()) rx = {{.*}}
auto ry = &unaryret<const volatile int*, 23>;
// CHECK: (const volatile int *(*)()) ry = {{.*}}
auto rz = &unaryret<void*&, 24>;
// CHECK: (void *&(*)()) rz = {{.*}}

// FIXME: This output doesn't really look correct.  It should probably be
// formatting this as `int(&)[5] (*)()`.
auto raa = &unaryret<int(&)[5], 25>;
// CHECK: (int (&(*)())[5]) raa = {{.*}}
auto rab = &unaryret<int(*)[5], 26>;
// CHECK: (int (*(*)())[5]) rab = {{.*}}
auto rac = &unaryret<int(&&)[5], 27>;
// CHECK: (int (&&(*)())[5]) rac = {{.*}}
auto rad = &unaryret<int(*const)[5], 28>;
// CHECK: (int (*const (*)())[5]) rad = {{.*}}



// Function references, we only need a couple of these since most of the
// interesting cases are already tested.
auto &ref = unary<bool>;
// CHECK: (void (&)(bool)) ref = {{.*}} (&::ref = <no summary available>)
auto &ref2 = unary<volatile int*>;
// CHECK: (void (&)(volatile int *)) ref2 = {{.*}} (&::ref2 = <no summary available>)
auto &ref3 = varargs;
// CHECK: (int (&)(int, int, ...)) ref3 = {{.*}} (&::ref3 = <no summary available>)

// Multiple arguments, as before, just something to make sure it works.
auto binp = &binary<int*, const int*>;
// CHECK: (void (*)(int *, const int *)) binp = {{.*}}
auto &binr = binary<int*, const int*>;
// CHECK: (void (&)(int *, const int *)) binr = {{.*}} (&::binr = <no summary available>)

// And finally, a function with no arguments.
auto null = &nullary;
// CHECK: (void (*)()) null = {{.*}}

// FIXME: These currently don't work because clang-cl emits incorrect debug info
// for std::nullptr_t.  We should fix these in clang-cl.
auto rae = &unaryret<decltype(nullptr), 29>;
// CHECK: (std::nullptr_t (*)()) rae = {{.*}}
auto aae = &unary<decltype(nullptr)>;
// CHECK: (void (*)(std::nullptr_t)) aae = {{.*}}

int main(int argc, char **argv) {
  return 0;
}

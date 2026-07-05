// clang-format off
// REQUIRES: lld, x86

// Test that we can display function signatures with class types.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: %lldb -f %t.exe -s \
// RUN:     %p/Inputs/function-types-classes.lldbinit | FileCheck %s

// This is just some unimportant helpers needed so that we can get reference and
// rvalue-reference types into return values.
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


template<typename R>
R nullary() { return MakeResult<R>::result(); }

template<typename R, typename A, typename B>
R three(A a, B b) { return MakeResult<R>::result(); }

template<typename R, typename A, typename B, typename C>
R four(A a, B b, C c) { return MakeResult<R>::result(); }

struct S {};
class C {};
union U {};
enum E {};

namespace A {
  namespace B {
    // NS::NS
    struct S { };
  }

  struct C {
    // NS::Struct
    struct S {};
  };
}

struct B {
  struct A {
    // Struct::Struct
    struct S {};
  };
};

// clang (incorrectly) doesn't emit debug information for outer classes
// unless they are instantiated.  They should also be emitted if there
// is an inner class which is instantiated.
A::C ForceInstantiateAC;
B ForceInstantiateB;
B::A ForceInstantiateBA;

template<typename T>
struct TC {};

// const and volatile modifiers
auto a = &four<S, C*, U&, E&&>;
// CHECK: (S (*)(C *, U &, E &&)) a = {{.*}}
auto b = &four<E, const S*, const C&, const U&&>;
// CHECK: (E (*)(const S *, const C &, const U &&)) b = {{.*}}
auto c = &four<U, volatile E*, volatile S&, volatile C&&>;
// CHECK: (U (*)(volatile E *, volatile S &, volatile C &&)) c = {{.*}}
auto d = &four<C, const volatile U*, const volatile E&, const volatile S&&>;
// CHECK: (C (*)(const volatile U *, const volatile E &, const volatile S &&)) d = {{.*}}

// classes nested in namespaces and inner classes

auto e = &three<A::B::S*, B::A::S*, A::C::S&>;
// CHECK: (A::B::S *(*)(B::A::S *, A::C::S &)) e = {{.*}}
auto f = &three<A::C::S&, A::B::S*, B::A::S*>;
// CHECK: (A::C::S &(*)(A::B::S *, B::A::S *)) f = {{.*}}
auto g = &three<B::A::S*, A::C::S&, A::B::S*>;
// CHECK: (B::A::S *(*)(A::C::S &, A::B::S *)) g = {{.*}}

// parameter types that are themselves template instantiations.
auto h = &four<TC<void>, TC<int>, TC<TC<int>>, TC<A::B::S>>;
// CHECK: (TC<void> (*)(TC<int>, TC<TC<int>>, TC<A::B::S>)) h = {{.*}}

auto i = &nullary<A::B::S>;
// CHECK: (A::B::S (*)()) i = {{.*}}


// Make sure we can handle types that don't have complete debug info.
struct Incomplete;
auto incomplete = &three<Incomplete*, Incomplete**, const Incomplete*>;
// CHECK: (Incomplete *(*)(Incomplete **, const Incomplete *)) incomplete = {{.*}}

// CHECK: TranslationUnitDecl {{.*}}
// CHECK: |-CXXRecordDecl {{.*}} class C
// CHECK: |-CXXRecordDecl {{.*}} union U
// CHECK: |-EnumDecl {{.*}} E
// CHECK: |-CXXRecordDecl {{.*}} struct S
// CHECK: |-VarDecl {{.*}} a 'S (*)(C *, U &, E &&)'
// CHECK: |-VarDecl {{.*}} b 'E (*)(const S *, const C &, const U &&)'
// CHECK: |-VarDecl {{.*}} c 'U (*)(volatile E *, volatile S &, volatile C &&)'
// CHECK: |-VarDecl {{.*}} d 'C (*)(const volatile U *, const volatile E &, const volatile S &&)'
// CHECK: |-CXXRecordDecl {{.*}} struct B
// CHECK: | `-CXXRecordDecl {{.*}} struct A
// CHECK: |   `-CXXRecordDecl {{.*}} struct S
// CHECK: |-NamespaceDecl {{.*}} A
// CHECK: | |-CXXRecordDecl {{.*}} struct C
// CHECK: | | `-CXXRecordDecl {{.*}} struct S
// CHECK: | `-NamespaceDecl {{.*}} B
// CHECK: |   `-CXXRecordDecl {{.*}} struct S
// CHECK: |-VarDecl {{.*}} e 'A::B::S *(*)(B::A::S *, A::C::S &)'
// CHECK: |-VarDecl {{.*}} f 'A::C::S &(*)(A::B::S *, B::A::S *)'
// CHECK: |-VarDecl {{.*}} g 'B::A::S *(*)(A::C::S &, A::B::S *)'
// CHECK: |-CXXRecordDecl {{.*}} struct TC<int>
// CHECK: |-CXXRecordDecl {{.*}} struct TC<TC<int>>
// CHECK: |-CXXRecordDecl {{.*}} struct TC<A::B::S>
// CHECK: |-CXXRecordDecl {{.*}} struct TC<void>
// CHECK: |-VarDecl {{.*}} h 'TC<void> (*)(TC<int>, TC<TC<int>>, TC<A::B::S>)'
// CHECK: |-VarDecl {{.*}} i 'A::B::S (*)()'
// CHECK: |-CXXRecordDecl {{.*}} struct Incomplete
// CHECK: `-VarDecl {{.*}} incomplete 'Incomplete *(*)(Incomplete **, const Incomplete *)'

int main(int argc, char **argv) {
  return 0;
}

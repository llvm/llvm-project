#include <stdbool.h>
#include <stdio.h>

struct ForwardDecl;

typedef int MyInt;

void populate(MyInt i);

typedef enum MyEnum {
  eOne = 1,
  eTwo =2,
  eThree = 3,
} MyEnumT;

typedef union MyUnion {
    MyInt i;
    const char* s;
} MyUnionT;

typedef struct MyNestedStruct {
  MyInt i;
  const char* s;
  volatile char c;
  char a[4];
  MyEnumT e;
  MyUnionT u;
  _Bool b;
} MyNestedStructT;

typedef struct MyStruct {
  MyNestedStructT n;
  void (*f)(int);
} MyStructT;

struct LargeStruct {
  char buffer[9000];
  int b;
};

struct RecursiveStruct {
  struct RecursiveStruct *n;
};

MyStructT foo;
struct ForwardDecl *forward;
struct LargeStruct bar;
struct RecursiveStruct ke;

void populate(MyInt i) {
  foo.n.i = i;
  foo.n.s = "foo";
  foo.n.c = 'c';
  foo.n.a[0] = 'a';
  foo.n.a[1] = 'b';
  foo.n.a[2] = 'c';
  foo.n.a[3] = 'd';
  foo.n.e = eOne;
  foo.n.b = false;
  foo.f = NULL;
  forward = NULL;
  bar.b = i;
  ke.n = NULL;
}

int main(int argc, char** argv) {
  populate(argc);
  printf("foo is at address: %p\n", (void*)&foo); // Break here
  return 0;
}

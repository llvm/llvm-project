// RUN: %llvmgcc -S -emit-llvm %s -o - | grep llvm.gcroot
// RUN: %llvmgcc -S -emit-llvm %s -o - | grep llvm.gcroot | count 6
// RUN: %llvmgcc -S -emit-llvm %s -o - | llvm-as

typedef struct foo_s
{
  int a;
} foo, __attribute__ ((gcroot)) *foo_p;

foo my_foo;

int alpha ()
{
  foo my_foo2 = my_foo;
  
  return my_foo2.a;
}

int bar (foo a)
{
  foo_p b;
  return b->a;
}

foo_p baz (foo_p a, foo_p b, foo_p *c)
{
  a = b = *c;
  return a;
}

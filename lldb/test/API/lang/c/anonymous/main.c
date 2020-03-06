#include <stdio.h>

struct anonymous_nest {
  struct {
    struct {
      int a;
      int b;
    }; // anonymous
    struct {
      int c;
      int d;
    } foo;
  }; // anonymous
};

struct anonymous_child {
  struct {
    struct {
      int a;
      int b;
    } grandchild;
    struct {
      int c;
      int d;
    } foo;
  }; // anonymous
};

struct anonymous_grandchild {
  struct {
    struct {
      int a;
      int b;
    }; // anonymous
    struct {
      int c;
      int d;
    } foo;
  } child;
};

int processor_nest (struct anonymous_nest *n)
{
  return n->foo.d + n->b; // Set breakpoint 0 here.
}

int processor_child (struct anonymous_child *c)
{
  return c->foo.d + c->grandchild.b; // Set breakpoint 1 here.
}

int processor_grandchild (struct anonymous_grandchild *g)
{
  return g->child.foo.d + g->child.b;
}



typedef struct {
    int dummy;
} type_y;

typedef struct {
    type_y y;
} type_z;



int main()
{
  struct anonymous_nest n = { 0, 2, 0, 4 };
  struct anonymous_child c = { 0, 2, 0, 4 };
  struct anonymous_grandchild g = { 0, 2, 0, 4 };
  type_z *pz = 0;
  type_z z = {{2}};

  printf("%d\n", processor_nest(&n));
  printf("%d\n", processor_child(&c));
  printf("%d\n", processor_grandchild(&g)); // Set breakpoint 2 here.

  return 0;
}

// Testcase for PR1242
// RUN: %llvmgcc -S %s -o - | grep datalayout | \
// RUN:    not grep {"\[Ee\]-p:\[36\]\[24\]:\[36\]\[24\]"}
// END.
#include <stdlib.h>
#define NDIM 3
#define BODY 01
typedef double vector[NDIM];
typedef struct bnode* bodyptr;
// { i16, double, [3 x double], i32, i32, [3 x double], [3 x double], [3 x
// double], double, \2 *, \2 * }
struct bnode {
  short int type;
  double mass;
  vector pos;
  int proc;
  int new_proc;
  vector vel;
  vector acc;
  vector new_acc;
  double phi;
  bodyptr next;
  bodyptr proc_next;
} body;

#define Type(x) ((x)->type)
#define Mass(x) ((x)->mass)
#define Pos(x)  ((x)->pos)
#define Proc(x) ((x)->proc)
#define New_Proc(x) ((x)->new_proc)
#define Vel(x)  ((x)->vel)
#define Acc(x)  ((x)->acc)
#define New_Acc(x)  ((x)->new_acc)
#define Phi(x)  ((x)->phi)
#define Next(x) ((x)->next)
#define Proc_Next(x) ((x)->proc_next)

bodyptr ubody_alloc(int p)
{ 
  register bodyptr tmp;
  tmp = (bodyptr)malloc(sizeof(body));

  Type(tmp) = BODY;
  Proc(tmp) = p;
  Proc_Next(tmp) = NULL;
  New_Proc(tmp) = p;
  return tmp;
}

int main(int argc, char** argv) {
  bodyptr b = ubody_alloc(17);
  return 0;
}

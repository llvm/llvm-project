// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/* GCC Used to generate code that contained a branch to the entry node of 
 * the do_merge function.  This is illegal LLVM code.  To fix this, GCC now
 * inserts an entry node regardless of whether or not it has to insert allocas.
 */

struct edge_rec
{
  struct VERTEX *v;
  struct edge_rec *next;
  int wasseen;
  int more_data;
};

typedef struct edge_rec *QUAD_EDGE;

typedef struct {
  QUAD_EDGE left, right;
} EDGE_PAIR;

struct EDGE_STACK {
    int ptr;
    QUAD_EDGE *elts;
    int stack_size;
};

int do_merge(QUAD_EDGE ldo, QUAD_EDGE rdo) {
  int lvalid;
  QUAD_EDGE basel,rcand;
  while (1) {
    if (!lvalid) {
      return (int)basel->next;
    }
  }
}


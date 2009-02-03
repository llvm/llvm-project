/* RUN: %llvmgcc %s -S -o - -emit-llvm -O0 | grep {zeroinitializer.*zeroinitializer.*zeroinitializer.*zeroinitializer.*zeroinitializer.*zeroinitializer}

The FE must generate padding here both at the end of each PyG_Head and
between array elements.  Reduced from Python. */

typedef union _gc_head {
  struct {
    union _gc_head *gc_next;
    union _gc_head *gc_prev;
    long gc_refs;
  } gc;
  int dummy __attribute__((aligned(16)));
} PyGC_Head;

struct gc_generation {
  PyGC_Head head;
  int threshold;
  int count;
};

#define GEN_HEAD(n) (&generations[n].head)

/* linked lists of container objects */
static struct gc_generation generations[3] = {
        /* PyGC_Head,                           threshold,      count */
        {{{GEN_HEAD(0), GEN_HEAD(0), 0}},       700,            0},
        {{{GEN_HEAD(1), GEN_HEAD(1), 0}},       10,             0},
        {{{GEN_HEAD(2), GEN_HEAD(2), 0}},       10,             0},
};

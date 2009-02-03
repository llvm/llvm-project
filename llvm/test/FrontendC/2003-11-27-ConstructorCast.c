// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

struct i387_soft_struct {
  long cwd;
};
union i387_union {
  struct i387_soft_struct soft;
};
struct thread_struct {
  union i387_union i387;
};
void _init_task_union(void) {
   struct thread_struct thread = (struct thread_struct) { {{0}} };
}

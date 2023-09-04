#include <sys/prctl.h>

// This is its own function so that lldb can call it.
int setup_mte() {
  return prctl(PR_SET_TAGGED_ADDR_CTRL, PR_TAGGED_ADDR_ENABLE | PR_MTE_TCF_SYNC,
               0, 0, 0);
}

int main(int argc, char const *argv[]) {
  if (setup_mte())
    return 1;

  return 0; // Set a break point here.
}

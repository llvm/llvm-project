#include "lib.h"

static void callee0() {}
void callee1() {}

typedef void (*FPT)(); 
FPT calleeAddrs[] = {callee0, callee1};

void global_func() {
    FPT fp = nullptr;
    for (int i = 0; i < 5; i++) {
      fp = calleeAddrs[i % 2];
      fp();
    }
}

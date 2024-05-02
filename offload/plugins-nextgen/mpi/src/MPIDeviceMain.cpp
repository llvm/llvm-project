#include "EventSystem.h"

int main(int argc, char *argv[]) {
  EventSystemTy EventSystem;

  EventSystem.initialize();

  EventSystem.runGateThread();

  EventSystem.deinitialize();
}

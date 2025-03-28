#define HIDE_FROM_PLUGIN 1
#include "service.h"

struct ServiceAux {
  Service *Owner;
};

struct Service::State {};

void exported() {
  // Make sure debug-info for definition of Service is
  // emitted in this CU.
  Service service;
  service.start(0);
}

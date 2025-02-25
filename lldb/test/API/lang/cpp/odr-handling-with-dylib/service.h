#ifndef SERVICE_H_IN
#define SERVICE_H_IN

struct ServiceAux;

struct Service {
  struct State;
  bool start(State *) { return true; }

#ifdef HIDE_FROM_PLUGIN
  int __resv1;
#endif // !HIDE_FROM_PLUGIN

  Service *__owner;
  ServiceAux *aux;
};

void exported();

#endif

void foo(void) {}

__attribute__((transparent_stepping))
void bar(void) {
  foo();
}

__attribute__((transparent_stepping))
void baz(void) {
  bar();
}

__attribute__((nodebug))
void nodebug(void) {}

__attribute__((transparent_stepping))
void nodebug_then_trampoline(void) {
  nodebug();
  baz();
}

__attribute__((transparent_stepping))
void doesnt_call_trampoline(void) {}

void direct_trampoline_call(void) {
  bar(); // Break here for direct 
  bar();
}

void chained_trampoline_call(void) {
  baz(); // Break here for chained
  baz();
}

void trampoline_after_nodebug(void) {
  nodebug_then_trampoline(); // Break here for nodebug then trampoline
  nodebug_then_trampoline();
}

void unused_target(void) {
  doesnt_call_trampoline(); // Break here for unused
}


int main(void) {
  direct_trampoline_call();
  chained_trampoline_call();
  trampoline_after_nodebug();
  unused_target();
  return 0;
}


#include <stdio.h>
extern void PrintSomething();
extern "C" {
void LLDBServerPluginInitialize() {
  puts("LLDBServerPluginInitialize");
  // PrintSomething();
}
}

#define MakeResolver(name)                                       \
  void * name ## Resolver(void) __asm__("_" #name);              \
  void * name ## Resolver(void) {                                \
    __asm__(".symbol_resolver _" #name);                         \
    return name ## _hidden;                                    \
  }

int 
call_through_indirect_hidden(int arg)
{
  return arg + 5;
}

MakeResolver(call_through_indirect)

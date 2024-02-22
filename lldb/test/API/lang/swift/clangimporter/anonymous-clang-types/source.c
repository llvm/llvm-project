#include "bridging-header.h"

TwoAnonymousStructs makeTwoAnonymousStructs() {
    TwoAnonymousStructs anon_struct;
    anon_struct.x = 1;
    anon_struct.y = 2;
    anon_struct.z = 3;
    anon_struct.a = 4;
    return anon_struct;
}


TwoAnonymousUnions makeTwoAnonymousUnions() {
  TwoAnonymousUnions anon_unions;
  anon_unions.y = 2;
  anon_unions.z = 3;
  anon_unions.a = 4;
  anon_unions.b = 5;
  anon_unions.c = 6;
  return anon_unions;
}

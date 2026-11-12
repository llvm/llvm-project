#include <stdlib.h>
#include <string.h>

struct WithFlexChar {
  int member;
  char flexible[];
};

struct WithFlexSChar {
  int member;
  signed char flexible[];
};

struct WithFlexUChar {
  int member;
  unsigned char flexible[];
};

#define CONTENTS "contents"

int main() {
  struct WithFlexChar *c =
      (struct WithFlexChar *)malloc(sizeof(int) + sizeof(CONTENTS));
  c->member = 1;
  strcpy(c->flexible, CONTENTS);

  struct WithFlexSChar *sc =
      (struct WithFlexSChar *)malloc(sizeof(int) + sizeof(CONTENTS));
  sc->member = 1;
  strcpy((char *)sc->flexible, CONTENTS);

  struct WithFlexUChar *uc =
      (struct WithFlexUChar *)malloc(sizeof(int) + sizeof(CONTENTS));
  uc->member = 1;
  strcpy((char *)uc->flexible, CONTENTS);
  return 0; // break here
}

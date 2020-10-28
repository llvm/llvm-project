#include "test.h"
#include <cstdlib>

bool on = false;
int len = 42;
double grid[10][10] = {0};
const char * hello = "world";
const char * world = "hello";

void A::B() {}

void useless() {}

double more_useless() {
  return 0;
}

int foo() {
  on = true;
  return 3;
}

int bar() {
  len--;
  return foo() + 45;
}

void assign(int ii, int jj) {
  grid[ii][jj] = (ii+1) * (jj+1);
}

void initialize_grid() {
  for (int ii = 0; ii < 2; ii++)
    for (int jj = 0; jj < 2; jj++)
      assign(ii, jj);
}

int main() {
  initialize_grid();

  int a = 2;
  on = rand() % 2;
  if (on) {
    foo();
    ++a;
  } else {
    bar();
    a += rand();
  }

  for (int ii = 0; ii < 10; ++ii) {
    switch (rand() % 5) {
      case 0:
        a += rand();
        break;
      case 1:
      case 2:
        a += rand() / rand();
        break;
      case 3:
        a -= rand();
        break;
      default:
        a = -1;
    }
  }

  A thing;
  for (uint64_t ii = 0; ii < 4294967296; ++ii)
    thing.B();

  return a + 8 + grid[2][3] + len;
  return more_useless();
}


#include <iostream>

using namespace std;

int next() {
  static int i = 0;
  cout << "incrementing " << i << endl;
  return ++i;
}

int main() {
  int i = 0;
  while (i < 5)
    i = next();
  return 0;
}

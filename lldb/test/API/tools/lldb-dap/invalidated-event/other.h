#ifndef OTHER_H
#define OTHER_H

int add(int a, int b) {
  int first = a;
  int second = b; // thread return breakpoint
  int result = first + second;
  return result;
}
#endif // OTHER_H

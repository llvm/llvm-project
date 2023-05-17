#include <cstdio>
#include <deque>
typedef std::deque<int> int_deq;

void by_ref_and_ptr(std::deque<int> &ref, std::deque<int> *ptr) {
  printf("stop here");
  return;
}

int main() {
  int_deq numbers;
  printf("break here");

  (numbers.push_back(1));
  printf("break here");

  (numbers.push_back(12));
  (numbers.push_back(123));
  (numbers.push_back(1234));
  (numbers.push_back(12345));
  (numbers.push_back(123456));
  (numbers.push_back(1234567));
  by_ref_and_ptr(numbers, &numbers);
  printf("break here");

  numbers.clear();
  printf("break here");

  return 0;
}

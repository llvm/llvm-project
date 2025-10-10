#include <stdio.h>
#include <string>
#include <vector>
typedef std::vector<int> int_vect;
typedef std::vector<std::string> string_vect;

template <class T>
void by_ref_and_ptr(std::vector<T> &ref, std::vector<T> *ptr) {
  // Stop here to check by ref
  return;
}

int main() {
  int_vect numbers;
  numbers.push_back(1);  // break here
  numbers.push_back(12); // break here
  numbers.push_back(123);
  numbers.push_back(1234);
  numbers.push_back(12345); // break here
  numbers.push_back(123456);
  numbers.push_back(1234567);
  by_ref_and_ptr(numbers, &numbers);

  printf("break here");
  numbers.clear();

  numbers.push_back(7); // break here

  string_vect strings;
  strings.push_back(std::string("goofy"));
  strings.push_back(std::string("is"));
  strings.push_back(std::string("smart"));
  printf("break here");
  strings.push_back(std::string("!!!"));

  printf("break here");
  strings.clear();

  return 0; // break here
}

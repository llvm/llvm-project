static int this_is_a_very_long_function_with_a_bunch_of_arguments(
    int first, int second, int third, int fourth, int fifth) {
  int result = first + second + third + fourth + fifth;
  return result;
}

int square(int x) { return x * x; }

int main(int argc, char const *argv[]) {
  // This is a random comment.
  int did_call = 0;

  int first = 1;
  int second = 2;
  int third = 3;
  int fourth = 4;
  int fifth = 5;

  //                                    v In the middle of a function name (col:42)
  int result = this_is_a_very_long_function_with_a_bunch_of_arguments(
      first, second, third, fourth, fifth);

  //                  v In the middle of the lambda declaration argument (col:23)
  auto lambda = [&](int n) {
  //                     v Inside the lambda (col:26)
    return first + third * n;
  };

  result = lambda(3);

  //                                             v At the beginning of a function name (col:50)
  if(square(argc+1) != 0) { did_call = 1; return square(argc); }

  return square(0);
}

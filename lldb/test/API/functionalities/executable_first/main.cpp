extern int b_function();

int main(int argc, char* argv[]) {
  int ret_value = b_function();
  return ret_value; // break after function call
}

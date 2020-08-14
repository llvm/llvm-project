int single_local_func() {
  // This function should always only have a single local variable and no
  // parameters.
  int only_local = 3;
  return only_local; // break in single_local_func
}

int main(int argc, char **argv) {
  int to_complete = 0;
  return to_complete + single_local_func();
}

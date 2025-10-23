void stop() {}

int main(int argc, char **argv) {
  int array[10];
  array[0] = 0;
  int (&array_ref)[10] = array;
  int *p_int0 = &array[0];

  stop(); // Set a breakpoint here
  return 0;
}

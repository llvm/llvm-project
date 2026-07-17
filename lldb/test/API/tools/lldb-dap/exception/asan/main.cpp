int main() {
  int *array = new int[100];
  delete[] array;
  return array[42]; // asan
}

int bad_read(int index) {
  int array[] = {0, 1, 2};
  return array[index];
}

int main() {
  bad_read(10);

  return 0;
}

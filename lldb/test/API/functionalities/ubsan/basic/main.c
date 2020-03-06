int main() {
  int data[4];
  return *(int *)(((char *)&data[0]) + 2); // align line
}

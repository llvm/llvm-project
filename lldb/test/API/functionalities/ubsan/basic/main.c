int main() {
  int data[4];
  int result = *(int *)(((char *)&data[0]) + 2); // align line

  int *p = data + 5;  // Index 5 out of bounds for type 'int [4]'
  *p = data + 5;
  *p = data + 5;
  *p = data + 5;
  *p = data + 5;
  *p = data + 5;
  *p = data + 5;
  *p = data + 5;
  *p = data + 5;

  return 0;
}

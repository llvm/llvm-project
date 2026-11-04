int main() {
  int data[4];

  char *addr = &data[0];
  int *ptr = addr + 2;
  int result = *ptr; // align line

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

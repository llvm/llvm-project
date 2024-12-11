int arr[10];
int v;

struct S { 
    int i;
    char *p; 
};

int f(int *p) {
  return *p; // break here
}

int* foo(int count) {
    int arr[10];
    int *local = &arr[0];
    for (int i = 0; i < count; ++i)
        local[i] = i;
    return local;
}

int main() {
  int x = *foo(5);

  int *ptr1 = &arr[v];
  int *ptr2 = arr;

  struct S s = {1, 0};
  struct S *ps = &s;

  f(ptr1); // break here

  ptr1++;

  return x;
}

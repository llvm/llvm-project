struct Foo {
  int real_child = 47;
};

struct HasFoo {
  Foo f;
};

struct Point {
  int x;
  int y;
};

int main() {
  Foo foo;
  HasFoo has_foo;
  Point point_arr[] = {{1, 2}, {3, 4}, {5, 6}};
  int int_arr[] = {1, 2, 3, 4, 5, 6};
  Point *point_ptr = point_arr;
  int *int_ptr = int_arr;
  return 0; // break here
}

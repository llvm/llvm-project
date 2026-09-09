int main(int argc, char const *argv[]) {
  struct {
    int a;
    int b;
    int c;
  } MyAggregateDataType = {1, 2, 3};

  return MyAggregateDataType.a; // break here
}

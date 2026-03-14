struct Struct {
  int x;
};

int main() {
  Struct use;
  use.x = 3;
  return use.x; // break here
}

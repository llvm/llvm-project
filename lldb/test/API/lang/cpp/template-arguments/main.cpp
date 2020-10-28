template<typename T, unsigned value>
struct C {
  T member = value;
};

C<int, 2> temp1;

int main() {}

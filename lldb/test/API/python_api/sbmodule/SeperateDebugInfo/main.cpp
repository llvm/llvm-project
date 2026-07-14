extern int bar(void);

int main() {
  int x = bar();
  x += 2; // break here
  return x;
}

class Empty {};

int main (int argc, char const *argv[]) {
  Empty e;
  Empty* ep = new Empty;
  return 0; // Break at this line
}

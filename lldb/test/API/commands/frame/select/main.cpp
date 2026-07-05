int nested3() {
  return 3; // Set break point at this line.
}

int nested2() {
  return 2 + nested3();
}

int nested1() {
  return 1 + nested2();
}


int main(int argc, char **argv) {
  return nested1();
}

void* foo(void *p)
{
  return p; // break here
}

int main() {
  while (1) {
    foo(0);
  }
  return 0;
}

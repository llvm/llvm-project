
struct mystruct {
  int c, d, e;
} global = {1, 2, 3};

int main ()
{
  return global.c; // break here
}

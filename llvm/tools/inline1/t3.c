int foo(int a, int b); 
int goo(int a, int b); 
int small(int);

int main() {
  int s = small(3);
  s =+ goo(3,2)* goo(5,1) * small(s*2);
  return foo(s, s+1)- small(s+3);
}

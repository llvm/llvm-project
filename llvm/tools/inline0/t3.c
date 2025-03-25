int foo(int a, int b); 
int goo(int a, int b); 

int main() {
  int s = 0;
  s =+ goo(3,2)* goo(5,1);
  return foo(s, s+1);
}

#include <functional>

int foo(int x, int y) {
  return x + y - 1;
}

struct Bar {
   int operator()() {
       return 66 ;
   }
   int add_num(int i) const { return i + 3 ; }
   int add_num2(int i) {
     std::function<int (int)> add_num2_f = [](int x) {
         return x+1;
      };

      return add_num2_f(i); // Set break point at this line.
   }
} ;

int foo2() {
   auto f = [](int x) {
       return x+1;
   };

   std::function<int (int)> foo2_f = f;

   return foo2_f(10); // Set break point at this line.
}

int main (int argc, char *argv[])
{
  int acc = 42;
  std::function<int (int,int)> f1 = foo;
  std::function<int (int)> f2 = [acc,f1] (int x) -> int {
    return x+f1(acc,x);
  };

  auto f = [](int x, int y) { return x + y; };
  auto g = [](int x, int y) { return x * y; } ;
  std::function<int (int,int)> f3 =  argc %2 ? f : g ;

  Bar bar1 ;
  std::function<int ()> f4( bar1 ) ;
  std::function<int (const Bar&, int)> f5 = &Bar::add_num;

  int foo2_result = foo2();
  int bar_add_num2_result = bar1.add_num2(10);

  return f1(acc,acc) + f2(acc) + f3(acc+1,acc+2) + f4() + f5(bar1, 10); // Set break point at this line.
}

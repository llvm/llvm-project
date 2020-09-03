#include <cstdlib>

int side_effect = 0;

struct B { int dummy = 2324; };
struct C {
  void *operator new(std::size_t size) { void *p = ::operator new(size); side_effect = 3; return p; }
  void *operator new[](std::size_t size) { void *p = ::operator new(size); side_effect = 4; return p; }
  void operator delete(void *p) { std::free(p); side_effect = 1; }
  void operator delete[](void *p) { std::free(p); side_effect = 2; }

  B b;
  B* operator->() { return &b; }
  int operator->*(int) { return 2; }
  int operator+(int) { return 44; }
  int operator+=(int) { return 42; }
  int operator++(int) { return 123; }
  int operator++() { return 1234; }
  int operator-(int) { return 34; }
  int operator-=(int) { return 32; }
  int operator--() { return 321; }
  int operator--(int) { return 4321; }

  int operator*(int) { return 51; }
  int operator*=(int) { return 52; }
  int operator%(int) { return 53; }
  int operator%=(int) { return 54; }
  int operator/(int) { return 55; }
  int operator/=(int) { return 56; }
  int operator^(int) { return 57; }
  int operator^=(int) { return 58; }

  int operator|(int) { return 61; }
  int operator|=(int) { return 62; }
  int operator||(int) { return 63; }
  int operator&(int) { return 64; }
  int operator&=(int) { return 65; }
  int operator&&(int) { return 66; }

  int operator~() { return 71; }
  int operator!() { return 72; }
  int operator!=(int) { return 73; }
  int operator=(int) { return 74; }
  int operator==(int) { return 75; }

  int operator<(int) { return 81; }
  int operator<<(int) { return 82; }
  int operator<=(int) { return 83; }
  int operator<<=(int) { return 84; }
  int operator>(int) { return 85; }
  int operator>>(int) { return 86; }
  int operator>=(int) { return 87; }
  int operator>>=(int) { return 88; }

  int operator,(int) { return 2012; }
  int operator&() { return 2013; }

  int operator()(int) { return 91; }
  int operator[](int) { return 92; }

  operator int() { return 11; }
  operator long() { return 12; }

  // Make sure this doesn't collide with
  // the real operator int.
  int operatorint() { return 13; }
  int operatornew() { return 14; }
};

int main(int argc, char **argv) {
  C c;
  int result = c->dummy;
  result = c->*4;
  result += c+1;
  result += c+=1;
  result += c++;
  result += ++c;
  result += c-1;
  result += c-=1;
  result += c--;
  result += --c;

  result += c * 4;
  result += c *= 4;
  result += c % 4;
  result += c %= 4;
  result += c / 4;
  result += c /= 4;
  result += c ^ 4;
  result += c ^= 4;

  result += c | 4;
  result += c |= 4;
  result += c || 4;
  result += c & 4;
  result += c &= 4;
  result += c && 4;

  result += ~c;
  result += !c;
  result += c!=1;
  result += c=2;
  result += c==2;

  result += c<2;
  result += c<<2;
  result += c<=2;
  result += c<<=2;
  result += c>2;
  result += c>>2;
  result += c>=2;
  result += c>>=2;

  result += (c , 2);
  result += &c;

  result += c(1);
  result += c[1];

  result += static_cast<int>(c);
  result += static_cast<long>(c);
  result += c.operatorint();
  result += c.operatornew();

  C *c2 = new C();
  C *c3 = new C[3];

  //% self.expect("expr c->dummy", endstr=" 2324\n")
  //% self.expect("expr c->*2", endstr=" 2\n")
  //% self.expect("expr c + 44", endstr=" 44\n")
  //% self.expect("expr c += 42", endstr=" 42\n")
  //% self.expect("expr c++", endstr=" 123\n")
  //% self.expect("expr ++c", endstr=" 1234\n")
  //% self.expect("expr c - 34", endstr=" 34\n")
  //% self.expect("expr c -= 32", endstr=" 32\n")
  //% self.expect("expr c--", endstr=" 4321\n")
  //% self.expect("expr --c", endstr=" 321\n")
  //% self.expect("expr c * 3", endstr=" 51\n")
  //% self.expect("expr c *= 3", endstr=" 52\n")
  //% self.expect("expr c % 3", endstr=" 53\n")
  //% self.expect("expr c %= 3", endstr=" 54\n")
  //% self.expect("expr c / 3", endstr=" 55\n")
  //% self.expect("expr c /= 3", endstr=" 56\n")
  //% self.expect("expr c ^ 3", endstr=" 57\n")
  //% self.expect("expr c ^= 3", endstr=" 58\n")
  //% self.expect("expr c | 3", endstr=" 61\n")
  //% self.expect("expr c |= 3", endstr=" 62\n")
  //% self.expect("expr c || 3", endstr=" 63\n")
  //% self.expect("expr c & 3", endstr=" 64\n")
  //% self.expect("expr c &= 3", endstr=" 65\n")
  //% self.expect("expr c && 3", endstr=" 66\n")
  //% self.expect("expr ~c", endstr=" 71\n")
  //% self.expect("expr !c", endstr=" 72\n")
  //% self.expect("expr c!=1", endstr=" 73\n")
  //% self.expect("expr c=1", endstr=" 74\n")
  //% self.expect("expr c==1", endstr=" 75\n")
  //% self.expect("expr c<1", endstr=" 81\n")
  //% self.expect("expr c<<1", endstr=" 82\n")
  //% self.expect("expr c<=1", endstr=" 83\n")
  //% self.expect("expr c<<=1", endstr=" 84\n")
  //% self.expect("expr c>1", endstr=" 85\n")
  //% self.expect("expr c>>1", endstr=" 86\n")
  //% self.expect("expr c>=1", endstr=" 87\n")
  //% self.expect("expr c>>=1", endstr=" 88\n")
  //% self.expect("expr c,1", endstr=" 2012\n")
  //% self.expect("expr &c", endstr=" 2013\n")
  //% self.expect("expr c(1)", endstr=" 91\n")
  //% self.expect("expr c[1]", endstr=" 92\n")
  //% self.expect("expr static_cast<int>(c)", endstr=" 11\n")
  //% self.expect("expr static_cast<long>(c)", endstr=" 12\n")
  //% self.expect("expr c.operatorint()", endstr=" 13\n")
  //% self.expect("expr c.operatornew()", endstr=" 14\n")
  //% self.expect("expr (new struct C); side_effect", endstr=" = 3\n")
  //% self.expect("expr (new struct C[1]); side_effect", endstr=" = 4\n")
  //% self.expect("expr delete c2; side_effect", endstr=" = 1\n")
  //% self.expect("expr delete[] c3; side_effect", endstr=" = 2\n")
  delete c2;
  delete[] c3;
  return 0;
}

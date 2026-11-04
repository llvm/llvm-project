struct Int {
  int i;
};
typedef Int Foo;
typedef Int *FooP;
typedef Foo Bar;
typedef Foo *BarP;

int main() {
  Int i = {42};
  Int *i_p = &i;
  Int **i_pp = &i_p;
  Int ***i_ppp = &i_pp;

  Foo f = i;
  Foo *f_p = &f;
  Foo **f_pp = &f_p;
  Foo ***f_ppp = &f_pp;

  FooP fp = f_p;
  FooP *fp_p = &fp;
  FooP **fp_pp = &fp_p;

  Bar b = i;
  Bar *b_p = &b;
  Bar **b_pp = &b_p;

  BarP bp = b_p;
  BarP *bp_p = &bp;
  BarP **bp_pp = &bp_p;
  return 0; // Set break point at this line.
}

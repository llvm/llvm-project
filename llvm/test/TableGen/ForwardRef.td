// RUN: llvm-tblgen %s -o -
// XFAIL: vg_leak

class bar {
  list<bar> x;
}

class foo;
class foo;

class baz { list<foo> y; }

class foo {

}


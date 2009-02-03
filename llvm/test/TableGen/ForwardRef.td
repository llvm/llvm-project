// RUN: tblgen %s -o -

class bar {
  list<bar> x;
}

class foo;
class foo;

class baz { list<foo> y; }

class foo {

}


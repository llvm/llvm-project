class test<code C> {
  code Code = C;
}

// TableGen used to allow code blocks starting in one file
// and finishing in the parent one.  This test checks
// that this is reported as an error from now on.
def foo : test<[{ hello world!

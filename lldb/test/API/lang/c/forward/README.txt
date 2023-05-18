This example has a function call in foo.c named "foo" that takes a forward
declaration to "struct bar" and uses it as a pointer argument. In main.c
we have a real declaration for "struct bar". We want to be able to find the
real definition of "struct bar" when we are stopped in foo in foo.c such that
when we stop in "foo" we see the contents of the "bar_ptr".

int foo(int);
int bar(void);
int display(void) { return foo(1) + bar(); }

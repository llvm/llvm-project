int foo(int *f);

int bar(int b) {
	int var = b + 1;
	return foo(&var);
}

int baz(int b) {
	return bar(b);
}

typedef int Foo;

int main() {
	int lval = 1;
	Foo* x = &lval;
	Foo& y = lval;
	Foo&& z = 1;
	return 0; // Set breakpoint here
}


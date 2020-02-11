enum Foo {
	Case1 = 1,
	Case2 = 2,
	Case45 = 45
};

int main() {
	Foo f = Case45;
	int x = 1;
	int y = 45;
	int z = 43;
	return 1; // Set break point at this line.
}

struct foo
{
	int X;
	int Y;
	foo(int a, int b) : X(a), Y(b) {}
};

int main()
{
	foo f(1,2);
	f.X = 4; // Set break point at this line.
	return 0;
}

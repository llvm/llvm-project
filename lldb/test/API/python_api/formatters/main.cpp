#include <stdio.h>
#include <vector>

struct JustAStruct
{
	int A;
	float B;
	char C;
	double D;
	long E;
	short F;
};

struct FooType
{
	int A;
	float B;
	char C;
	double D;
	long E;
	short F;
};

struct CCC
{
	int a, b, c;
};

struct Empty1 { void *data; };
struct Empty2 { void *data; };


int main(int argc, char const *argv[]) {
	JustAStruct foo;
	foo.A = 1;
	foo.B = 3.14;
	foo.C = 'e';
	foo.D = 6.28;
	foo.E = 3100419850;
	foo.F = 0;

	FooType bar;
	bar.A = 1;
	bar.B = 3.14;
	bar.C = 'e';
	bar.D = 6.28;
	bar.E = 3100419850;
	bar.F = 0;
	JustAStruct* foo_ptr = &foo;

	std::vector<int> int_vector;

	CCC ccc = {111, 222, 333};

        Empty1 e1;
        Empty2 e2;

	return 0; // Set break point at this line.
}

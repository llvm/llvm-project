namespace NN
{
  int a = 778899;
  int b = 665544;
  int c = 445566;
}

class A
{
public:
  A();
  int Method(int a, int b);

private:
  int a, b;
};

A::A() : a(10), b(100) { }

int a = 112233;
int b = 445566;
int c = 778899;

int
A::Method(int a, int b)
{
    {
        int a = 12345;
        int b = 54321;
        int c = 34567;
        this->a = a + b + this->b; // Break 2
    }

    {
        using namespace NN;
        int a = 10001;
        int b = 10002;
        int c = 10003;
        this->a = a + b + this->b; // Break 3
    }

    return this->a + this->b + a + b; // Break 4
}

int
Function(int a, int b)
{
    int A;

    {
        int a = 12345;
        int b = 54321;
        int c = 34567;
        A = a + b + c; // Break 5
    }

    {
        using namespace NN;
        int a = 10001;
        int b = 10002;
        int c = 10003;
        A = a + b + c; // Break 6
    }

    return A + a + b; // Break 7
}

int
main()
{
    A obj;
    return obj.Method(1, 2) + Function(1, 2); // Break 1
}
